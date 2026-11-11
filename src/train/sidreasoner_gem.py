"""
SIDReasoner SFT 的 GEM 损失变体(Stage 1/2 共用组件)

GEM 出自 ICLR 2025 论文 "Preserving Diversity in Supervised Fine-tuning
of Large Language Models"(Li et al., arXiv:2408.16673),
官方实现: https://github.com/liziniu/GEM

核心思想: 用带熵正则的逆 KL 分布匹配损失替换 CE。训练时以
q = softmax(logits / beta).detach() 作为固定目标分布, 梯度只经过
gene_log_probs = log_softmax(logits) 流动; beta -> 1 时退化为 CE 的
形式, beta 越小越保留(鼓励)输出多样性。官方论文/脚本默认 beta=0.7。

本文件逐行移植官方 sft_trainer_v2.py(适配 transformers>=4.46 的
版本, 与本仓库环境一致)与 train.py 的 TrainingArguments 扩展:
- gem_loss / compute_loss / compute_training_logs /
  chunked_entropy_from_logits 均与官方对齐
- 评估期间(control.should_evaluate)与 loss="ce" 时回退模型自带的
  CE 损失 -> eval_loss / 早停 / extra_eval 集与 CE 基线完全可比
- 训练日志附带 ce_loss 诊断量(官方 README: 无论用 CE 还是 GEM,
  ce_loss 都应随训练下降); print_entropy=True 时附带 token 熵
与官方的两处有意差异(为迎合 SIDReasoner 而非随意修改):
1. 官方 train.py 将 adam_beta2 默认改为 0.95, 此处保持 HF 默认,
   保证与 SIDReasoner CE 基线唯一的差别就是损失函数;
2. 官方通过覆写 _maybe_log_save_evaluate 注入训练日志, 该方法签名
   在 transformers 5.x 已变化, 此处改为覆写 log() 注入同样字段;
3. 未移植官方 utils/ 的 Triton 版损失(为大规模模型准备的加速版,
   参考实现即本文件的纯 PyTorch 版, Qwen3-1.7B 规模无必要)。
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import transformers
from transformers.trainer import (
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)

from src.train.sidreasoner_common import MultiEvalTrainer


@dataclass
class GemTrainingArguments(transformers.TrainingArguments):
    """官方 train.py 中 TrainingArguments 的 GEM 扩展(字段名一致)"""

    loss: str = field(
        default="gem",
        metadata={
            "help": "Loss name",
            "choices": ["gem", "ce"],
        },
    )
    gem_beta: float = field(
        default=0.7,
        metadata={
            "help": (
                "Hyper-parameter in GEM. A value between 0 and 1. A value "
                "close to 1.0 makes GEM behave more like CE, while a value "
                "close to 0.0 preserves more diversity."
            ),
        },
    )
    gem_h: str = field(
        default="linear",
        metadata={
            "help": (
                "Function $h$ in GEM. The 'logsigmoid' function is more "
                "adaptive, but the difference between 'logsigmoid' and "
                "'linear' is usually negligible."
            ),
            "choices": ["logsigmoid", "linear"],
        },
    )
    print_entropy: bool = field(
        default=False,
        metadata={"help": "Print entropy during training"},
    )


def gem_loss(logits, labels, num_items_in_batch=None, beta=0.7, ignore_index=-100, h="logsigmoid"):
    """
    GEM 损失(官方 sft_trainer_v2.py 的 GemSFTTrainer.gem_loss 原样移植)。

    labels 中 ignore_index(-100) 的位置不参与损失; num_items_in_batch
    非 None 时按官方方式对整个梯度累积宏批次做 token 数归一化。
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    mask = shift_labels != -100
    shift_logits = shift_logits[mask]
    shift_labels = shift_labels[mask]

    with torch.no_grad():
        logits_on_labels = torch.gather(
            shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        logits_diff = shift_logits - logits_on_labels.unsqueeze(-1)
        if h == "linear":
            weights = torch.ones_like(logits_diff)
        elif h == "logsigmoid":
            weights = F.sigmoid(0.01 * logits_diff)
        else:
            raise ValueError(h)

    gene_log_probs = F.log_softmax(shift_logits, dim=-1)
    q_probs = torch.exp(F.log_softmax(shift_logits / beta, dim=-1)).detach()

    real_log_probs = torch.gather(
        gene_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
    )

    if num_items_in_batch is not None:
        loss = -torch.sum(
            q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
        ).sum() / num_items_in_batch
    else:
        loss = -torch.sum(
            q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
        ).mean()

    return loss


class GemMultiEvalTrainer(MultiEvalTrainer):
    """
    MultiEvalTrainer 的 GEM 版: 训练用 GEM 损失, 评估(与 CE 模式)用
    模型自带 CE; 其余评估/早停/多评估集行为继承自 MultiEvalTrainer。
    """

    @torch.no_grad
    def compute_training_logs(self, logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        training_logs = {}
        if self.args.print_entropy:
            entropy = chunked_entropy_from_logits(
                shift_logits,
                batch_size=max(1, shift_logits.size(0) // 4),
            ).mean()
            training_logs["entropy"] = round(entropy.item(), 2)

        return training_logs

    def gem_loss(self, logits, labels, num_items_in_batch=None, beta=0.7, ignore_index=-100, h="logsigmoid"):
        return gem_loss(
            logits,
            labels,
            num_items_in_batch=num_items_in_batch,
            beta=beta,
            ignore_index=ignore_index,
            h=h,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        官方 sft_trainer_v2.py 的 compute_loss 移植(去掉 gem_triton 分支)。
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        # (官方基于 transformers>=4.46; 5.x 已移除 args.past_index, 兼容处理)
        if getattr(self.args, "past_index", -1) >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # 评估或 loss="ce" 时用模型自带的 CE, 保证 eval_loss 与基线可比
            if self.args.loss == "ce" or self.control.should_evaluate:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            else:
                loss = self.gem_loss(
                    outputs.logits,
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch,
                    beta=self.args.gem_beta,
                    h=self.args.gem_h,
                )

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        # 训练期间记录 ce_loss(与可选的 entropy), 供监控训练是否正常
        if not self.control.should_evaluate:
            self.training_logs = self.compute_training_logs(
                outputs.logits, inputs["labels"]
            )
            self.training_logs["ce_loss"] = (
                outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            )
            self.training_logs["ce_loss"] = round(self.training_logs["ce_loss"].item(), 4)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        """把 training_logs(ce_loss/entropy)并入周期性训练日志。

        官方在 _maybe_log_save_evaluate 中注入, 该私有方法签名随
        transformers 版本漂移; 覆写 log() 可在不同版本下达到同样效果。
        """
        if not self.control.should_evaluate and getattr(self, "training_logs", None):
            training_logs = self.training_logs
            self.training_logs = {}
            logs = {**training_logs, **logs}
        return super().log(logs, *args, **kwargs)


def chunked_entropy_from_logits(chunk_logits, batch_size=None):
    """
    Compute entropy from logits in a memory-efficient manner by introducing a batch_size parameter.

    Args:
        chunk_logits (torch.Tensor): Logits tensor of shape (total_samples, num_classes).
        batch_size (int): Number of samples to process per batch.

    Returns:
        torch.Tensor: Entropy tensor of shape (total_samples,).
    """
    total_samples, num_classes = chunk_logits.shape
    entropy_list = []
    if batch_size is None:
        batch_size = total_samples

    # Process logits in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        logits_batch = chunk_logits[start_idx:end_idx]  # Get a batch of logits

        # Compute logsumexp for the current batch
        logsumexp_batch = torch.logsumexp(logits_batch, dim=-1, keepdim=False)  # Shape: (batch_size,)
        # Compute probabilities in log-space without computing softmax
        normalized_logits = logits_batch - logsumexp_batch.unsqueeze(-1)       # Shape: (batch_size, num_classes)
        exp_normalized_logits = torch.exp(normalized_logits)                   # Shape: (batch_size, num_classes)
        # Compute entropy for the batch
        entropy_batch = logsumexp_batch - (logits_batch * exp_normalized_logits).sum(dim=-1)  # Shape: (batch_size,)

        entropy_list.append(entropy_batch)  # Store results for batch

    # Concatenate results from all batches
    if len(entropy_list) > 0:
        return torch.cat(entropy_list, dim=0)
    else:
        return torch.tensor(0.0)
