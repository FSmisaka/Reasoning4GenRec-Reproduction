"""
SIDReasoner 三阶段训练管线的共享组件

Stage 1(SID-语言对齐 SFT): src/train/sidreasoner_sft.py
Stage 2(推理激活 SFT):     src/train/sidreasoner_activation.py
Stage 3(GRPO 强化学习):    scripts/sidreasoner/rl.sh(基于 SIDReasoner
                           官方 verl fork, 见 README)

本文件迁移自 SIDReasoner 官方仓库的 sft_Qwen3.py /
sft_reasoning_activation.py 中的公共部分(MultiEvalTrainer、
DDP 装配、HF Trainer 配置), 保持训练行为与原始实现一致。
"""

import os
import random
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch
import transformers
from datasets import Dataset as HFDataset
from transformers import EarlyStoppingCallback

from src.models.sidreasoner import (
    load_sidreasoner_model,
    maybe_restrict_to_new_tokens,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiEvalTrainer(transformers.Trainer):
    """
    迁移自 SIDReasoner: 除主评估集(SID 预测)外, 每个 epoch 额外
    评估 title2sid / sid2title 翻译集的 loss(不触发早停回调)。
    """

    def __init__(self, *args, extra_eval_sets: Optional[Dict[str, HFDataset]] = None, **kwargs):
        self.extra_eval_sets = extra_eval_sets or {}
        super().__init__(*args, **kwargs)

    @contextmanager
    def _disable_callback(self, callback_cls):
        callbacks = self.callback_handler.callbacks
        removed = [cb for cb in callbacks if isinstance(cb, callback_cls)]
        if not removed:
            yield
            return
        self.callback_handler.callbacks = [
            cb for cb in callbacks if not isinstance(cb, callback_cls)
        ]
        try:
            yield
        finally:
            self.callback_handler.callbacks = callbacks

    def evaluate(
        self,
        eval_dataset: Optional[HFDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if not self.extra_eval_sets:
            return metrics
        for name, dataset in self.extra_eval_sets.items():
            if dataset is None:
                continue
            with self._disable_callback(EarlyStoppingCallback):
                extra_metrics = super().evaluate(
                    eval_dataset=dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{name}",
                )
            self.log(extra_metrics)
            metrics.update(extra_metrics)
        return metrics


def _decode_tokens(tokens, tokenizer_ref):
    if not isinstance(tokens, (list, tuple)):
        return ""
    valid_ids = [tid for tid in tokens if isinstance(tid, int) and tid >= 0]
    if not valid_ids:
        return ""
    return tokenizer_ref.decode(valid_ids, skip_special_tokens=False)


def preview_dataset(dataset, name, tokenizer_ref, max_samples=3):
    print(f"[Preview] {name}: displaying up to {max_samples} samples")
    preview_count = min(max_samples, len(dataset))
    for idx in range(preview_count):
        sample = dataset[idx]
        if not isinstance(sample, dict):
            continue
        if "input_ids" in sample:
            input_text = _decode_tokens(sample["input_ids"], tokenizer_ref)
            if input_text:
                print(f"Sample {idx + 1}:\n  Input : {input_text}")
        if "labels" in sample:
            label_ids = [
                tid for tid in sample["labels"] if isinstance(tid, int) and tid >= 0
            ]
            if label_ids:
                label_text = _decode_tokens(label_ids, tokenizer_ref)
                print(f"  Label : {label_text}")
                print(f"  Length: {len(label_ids)} tokens")
        print()


def to_hf_dataset(torch_dataset):
    return HFDataset.from_dict(
        {
            k: [v[k] for v in torch_dataset]
            for k in torch_dataset[0].keys()
        }
    )


def run_sft(
    stage_name,
    cfg,
    category,
    base_model,
    output_dir,
    tokenizer,
    num_new_tokens,
    train_dataset,
    val_data_sid_prediction,
    val_data_title2sid,
    val_data_sid2title,
):
    """
    共享 SFT 驱动(Stage 1/2 通用), 训练配置与原始实现一致:
    bf16 全参微调, epoch 级评估/保存, eval_loss 早停, 结束保存
    final_checkpoint(模型+tokenizer)。tokenizer 须为已扩展 SID
    token 的版本, 且与构造训练数据所用的是同一个实例。
    """
    train_cfg = cfg["training"][stage_name]
    seed = train_cfg["seed"]
    set_seed(seed)
    if train_cfg.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = train_cfg["wandb_project"]

    model, num_new_tokens = load_sidreasoner_model(
        base_model, tokenizer, num_new_tokens
    )
    if train_cfg.get("train_new_token_embeddings_only"):
        model = maybe_restrict_to_new_tokens(model, num_new_tokens)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Trainable parameters: {trainable_params} / {total_params} "
        f"({trainable_params / max(total_params, 1) * 100:.4f}%)"
    )

    main_rank = int(os.environ.get("RANK", 0)) == 0 and int(
        os.environ.get("LOCAL_RANK", 0)
    ) == 0
    if main_rank:
        preview_dataset(train_dataset, stage_name, tokenizer)

    hf_train_dataset = to_hf_dataset(train_dataset).shuffle(seed=42)
    hf_val_dataset = to_hf_dataset(val_data_sid_prediction).shuffle(seed=42)
    hf_title2sid = to_hf_dataset(val_data_title2sid).shuffle(seed=42)
    hf_sid2title = to_hf_dataset(val_data_sid2title).shuffle(seed=42)

    batch_size = train_cfg["batch_size"]
    micro_batch_size = train_cfg["micro_batch_size"]
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = (
            gradient_accumulation_steps // world_size
        )
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = MultiEvalTrainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        extra_eval_sets={
            "title2sid": hf_title2sid,
            "sid2title": hf_sid2title,
        },
        args=transformers.TrainingArguments(
            run_name=train_cfg.get("wandb_run_name")
            or f"{category}_{stage_name}",
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=train_cfg["warmup_steps"],
            num_train_epochs=train_cfg["epochs"],
            learning_rate=train_cfg["lr"],
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir=output_dir,
            save_total_limit=train_cfg["save_total_limit"],
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=False,
            report_to=train_cfg.get("report_to", "wandb"),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg["patience"]
            )
        ],
    )
    model.config.use_cache = False

    trainer.train(
        resume_from_checkpoint=train_cfg.get("resume_from_checkpoint")
    )

    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[{stage_name}] final checkpoint saved to {final_dir}")
    return final_dir
