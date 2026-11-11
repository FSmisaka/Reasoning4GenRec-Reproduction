"""
GEM 损失移植正确性验证(src/train/sidreasoner_gem.py)

对照官方 sft_trainer_v2.py 的公式手工实现参考值, 校验:
1. h="linear"/"logsigmoid" 与 beta 取值下的数值一致性
2. beta=1, h="linear" 时 loss == CE - H(p)(论文: beta->1 退化为 CE 形式)
3. num_items_in_batch 归一化与 .mean() 版本一致
4. -100 掩码位置不参与损失
"""

import sys

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from src.train.sidreasoner_gem import (
    GemTrainingArguments,
    chunked_entropy_from_logits,
    gem_loss,
)


def reference_gem(shift_logits, shift_labels, beta, h):
    """官方公式的独立实现(不经过被测函数)"""
    lp = F.log_softmax(shift_logits, dim=-1)
    q = F.softmax(shift_logits / beta, dim=-1).detach()
    lpy = torch.gather(lp, -1, shift_labels.unsqueeze(-1))
    with torch.no_grad():
        diff = shift_logits - torch.gather(
            shift_logits, -1, shift_labels.unsqueeze(-1)
        )
        w = torch.ones_like(diff) if h == "linear" else torch.sigmoid(0.01 * diff)
    return -(q * w * (lpy - lp)).sum(-1).mean()


def main():
    torch.manual_seed(0)
    B, S, V = 2, 20, 64
    logits = torch.randn(B, S, V, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, V, (B, S))
    labels[:, :6] = -100  # 模拟 prompt 掩码

    shift_logits = logits[:, :-1, :][labels[:, 1:] != -100]
    shift_labels = labels[:, 1:][labels[:, 1:] != -100]
    n_tok = shift_labels.numel()

    for beta in (0.3, 0.7, 1.0):
        for h in ("linear", "logsigmoid"):
            got = gem_loss(logits, labels, beta=beta, h=h)
            ref = reference_gem(shift_logits, shift_labels, beta, h)
            assert torch.allclose(got, ref, atol=1e-8), (beta, h, got, ref)
    print("[1] linear/logsigmoid x beta 与官方公式参考值一致")

    # beta=1, h=linear: loss = E_q[log p_j - log p_y] = CE - H(p), q = p
    got = gem_loss(logits, labels, beta=1.0, h="linear")
    lp = F.log_softmax(shift_logits, dim=-1)
    ce = F.nll_loss(lp, shift_labels)
    ent = -(lp * lp.exp()).sum(-1).mean()
    assert torch.allclose(got, ce - ent, atol=1e-8), (got, ce - ent)
    print(f"[2] beta=1 退化为 CE - H(p): {got.item():.6f} == "
          f"{ce.item():.6f} - {ent.item():.6f}")

    # num_items_in_batch = token 数时与 mean 版本一致
    got = gem_loss(logits, labels, num_items_in_batch=n_tok, beta=0.7)
    ref = gem_loss(logits, labels, beta=0.7)
    assert torch.allclose(got, ref, atol=1e-8)

    # 梯度有限且掩码位置不影响损失/梯度
    loss = gem_loss(logits, labels, beta=0.7, h="logsigmoid")
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    grad_masked = logits.grad[:, :5].clone()  # 前 5 个位置 label=-100(无梯度路径)
    perturbed = logits.detach().clone()
    perturbed[:, :5] += 10.0  # 扰动被掩码的 prompt 位置
    logits2 = perturbed.requires_grad_(True)
    loss2 = gem_loss(logits2, labels, beta=0.7, h="logsigmoid")
    loss2.backward()
    assert torch.allclose(loss2, loss, atol=1e-8)
    assert torch.allclose(logits2.grad[:, :5], grad_masked, atol=1e-8)
    print("[3] num_items 归一化 / -100 掩码 / 梯度检查通过")

    # fp32 输入(accelerate bf16 混合精度的 forward 输出)会被 cast 到 bf16,
    # 结果与直接给 bf16 输入一致(确定性 round), 且梯度可回流
    bf16_logits = logits.detach().to(torch.bfloat16).requires_grad_(True)
    fp32_logits = logits.detach().to(torch.float32).requires_grad_(True)
    l_bf = gem_loss(bf16_logits, labels, beta=0.7, h="logsigmoid")
    l_f32 = gem_loss(fp32_logits, labels, beta=0.7, h="logsigmoid")
    assert torch.allclose(l_f32, l_bf, atol=1e-4), (l_f32.item(), l_bf.item())
    l_f32.backward()
    assert torch.isfinite(fp32_logits.grad).all()
    print("[3b] fp32 输入自动转 bf16(与 accelerate 混合精度输出对接)检查通过")

    # entropy 工具: 与直接按定义计算一致
    e1 = chunked_entropy_from_logits(shift_logits, batch_size=7)
    e2 = -(lp * lp.exp()).sum(-1)
    assert torch.allclose(e1, e2, atol=1e-8)
    print("[4] chunked_entropy_from_logits 一致")

    # GemTrainingArguments 可携带 GEM 字段构造(字段名与官方 train.py 一致)
    args = GemTrainingArguments(
        output_dir="/tmp/sidreasoner_gem_test",
        loss="gem", gem_beta=0.7, gem_h="logsigmoid", print_entropy=True,
        report_to=[],
    )
    assert args.loss == "gem" and args.gem_beta == 0.7
    print("[5] GemTrainingArguments 构造通过")

    print("ALL GEM LOSS TESTS PASSED")


if __name__ == "__main__":
    main()
