"""SASRec Hidden State 用户编码器。

复用本仓库 SASRec 复现代码(src/models/sasrec.py)与其已训练
checkpoint(runs/<Category>_sasrec_*/best.pt, 结构见 src/train/sasrec.py)。

要点(与训练/评估管线严格一致):
- 物品 id 0 起始, 模型输入 +1(0 为 padding);
- 序列**左侧补零**(src/train/sasrec.py build_batches: ``seq[i, max_len-len:]``),
  causal self-attention 屏蔽 padding;
- 用户向量取最后一个有效交互位置(即左 padding 下的位置 -1)的 hidden
  state, 等价于 SASRec 推荐时的最终行为表征(full_scores 用 x[:, -1]);
- 历史超过 max_seq_len 时只保留**最近的** max_seq_len 条(与评测一致);
- dropout 处于 eval 模式, no_grad 前向;
- 输出 L2 归一化, 余弦相似度 = 内积。

无 Data Leakage: 输入只含历史物品, 不含目标物品; 目标物品在 RL 数据
里是 reward 的 ground truth, 不进入编码器。
"""

import numpy as np
import torch

from .base import BaseUserEncoder
from .registry import register_user_encoder


@register_user_encoder("sasrec")
class SASRecHiddenStateEncoder(BaseUserEncoder):
    """u_q = SASRec(H_q) 最后一个有效位置的 hidden state。

    Args:
        checkpoint: runs/<Category>_sasrec_*/best.pt 路径。checkpoint 内
            含 ``state_dict`` / ``config`` / ``n_items``; 模型超参优先从
            state_dict 形状反推(hidden/max_seq_len/n_items), 与训练配置
            无关地稳健重建。
        device: "cuda" | "mps" | "cpu" | None(自动)。
    """

    name = "sasrec"

    def __init__(self, checkpoint: str, device: str = None):
        from src.models.sasrec import SASRec

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if "item_embedding.weight" not in state_dict:
            raise ValueError(f"{checkpoint} 中没有 item_embedding.weight, 不是 SASRec checkpoint。")

        # 模型超参: 优先 shape/键名反推(与训练配置解耦), config 补充无法
        # 从权重推断的项(n_heads / d_inner / dropout 等, 推理仅 n_heads 生效)
        n_items = int(state_dict["item_embedding.weight"].shape[0]) - 1
        hidden = int(state_dict["item_embedding.weight"].shape[1])
        max_seq_len = int(state_dict["position_embedding.weight"].shape[0])
        n_layers = 1 + max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")
        )
        model_cfg = {}
        if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
            model_cfg = ckpt["config"].get("model", {}) or {}
        n_heads = int(model_cfg.get("n_heads", 1))
        self._n_items = n_items
        self._max_seq_len = max_seq_len

        model = SASRec(
            n_items=n_items,
            max_seq_len=max_seq_len,
            hidden=hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            d_inner=int(model_cfg.get("d_inner", 4 * hidden)),
            dropout=0.0,
        )
        # 训练保存的是完整 state_dict(含 item_embedding/padding), 直接加载
        model.load_state_dict(state_dict)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def dim(self) -> int:
        return int(self.model.item_embedding.embedding_dim)

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def _collate(self, histories: list) -> torch.Tensor:
        """历史 -> 左 padding 的 (N, max_seq_len) long 张量(1 起始 id)。

        超长历史保留最近 max_seq_len 条(与 SASRec 滑窗评测语义一致)。
        """
        seq = torch.zeros(len(histories), self._max_seq_len, dtype=torch.long)
        for i, hist in enumerate(histories):
            if hist is None or len(hist) == 0:
                continue
            ids = [int(x) + 1 for x in hist][-self._max_seq_len :]
            seq[i, self._max_seq_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)
        return seq

    @torch.no_grad()
    def encode_histories(self, histories: list) -> np.ndarray:
        empty = [h is None or len(h) == 0 for h in histories]
        seq = self._collate(histories).to(self.device)
        hidden = self.model.last_hidden(seq)  # (N, hidden), 最后一有效位置
        hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        if any(empty):
            # 空历史无行为信号: 置零(全 padding 行经 softmax 会产生
            # 无意义的均匀混合向量, 不能当作用户表征)
            mask = torch.as_tensor(empty, dtype=torch.bool, device=hidden.device)
            hidden = hidden.masked_fill(mask[:, None], 0.0)
        return hidden.float().cpu().numpy()
