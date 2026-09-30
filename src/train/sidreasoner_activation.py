"""
SIDReasoner Stage 2: 推理激活 SFT

迁移自 SIDReasoner 官方仓库的 sft_reasoning_activation.py +
sft_reasoning_activation.sh。用 Stage 1 产出的 checkpoint 继续训练,
数据为 integrated_narrative.csv 中教师模型生成的"先推理后推荐"样本,
对应论文 3.3.1 节 Cold-start Reasoning Activation(仅训练一个 epoch
量级的轻量阶段, 目的是稳定"先 reasoning 再输出 SID"的响应格式)。

启动(4 卡示例):
  DATASET=games .venv/bin/python -m torch.distributed.run \
      --nproc_per_node 4 --master_port 29519 -m src.train.sidreasoner_activation
或 scripts/sidreasoner/activation.sh / make games-sidreasoner-activation
"""

import os

from config import load_config
from src.data.sidreasoner_data import (
    ReasoningActivationDataset,
    SidItemFeatDataset,
    SidSFTDataset,
)
from src.models.sidreasoner import (
    category_phrase,
    load_sidreasoner_tokenizer,
)
from src.train.common import resolve_category
from src.train.sidreasoner_common import run_sft


def main(cfg=None):
    cfg = cfg or load_config("sidreasoner")
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    train_cfg = cfg["training"]["activation"]
    category = resolve_category(data_cfg)
    phrase = category_phrase(category)

    root = data_cfg["root"]
    fname = f"{category}_5_2016-10-2018-11.csv"
    paths = {
        "eval_file": os.path.join(root, "valid", fname),
        "index_file": os.path.join(root, "index", f"{category}.index.json"),
        "item_meta": os.path.join(root, "index", f"{category}.item.json"),
        "narrative": os.path.join(
            root, "index", f"{category}.integrated_narrative.csv"
        ),
    }
    stage1_out = cfg["training"]["sft"]["out"] or os.path.join(
        "runs", "sidreasoner", f"{category}_stage1_sft"
    )
    base_model = train_cfg.get("base_model") or os.path.join(
        stage1_out, "final_checkpoint"
    )

    # Stage 2 的 base_model 是 Stage 1 的 final_checkpoint,
    # 其词表已含 SID token, load_sidreasoner_tokenizer 为幂等操作
    tokenizer, num_new_tokens = load_sidreasoner_tokenizer(
        base_model, paths["index_file"]
    )

    max_len = train_cfg["cutoff_len"]
    seed = train_cfg["seed"]
    sample = train_cfg.get("sample", -1)

    train_data = ReasoningActivationDataset(
        reasoning_train_file=paths["narrative"],
        item_file=paths["item_meta"],
        index_file=paths["index_file"],
        tokenizer=tokenizer,
        max_len=max_len,
        sample=sample,
        seed=seed,
        category=phrase,
    )
    val_sid = SidSFTDataset(
        train_file=os.path.join(root, "valid", fname), tokenizer=tokenizer,
        max_len=max_len, sample=sample, seed=seed, category=phrase,
        test=False, mask_assistant=True,
    )
    val_title2sid = SidItemFeatDataset(
        item_file=paths["item_meta"], index_file=paths["index_file"],
        tokenizer=tokenizer, max_len=max_len, sample=sample, seed=seed,
        category=phrase, task_type="title2sid", test=False,
        mask_assistant=True,
    )
    val_sid2title = SidItemFeatDataset(
        item_file=paths["item_meta"], index_file=paths["index_file"],
        tokenizer=tokenizer, max_len=max_len, sample=sample, seed=seed,
        category=phrase, task_type="sid2title", test=False,
        mask_assistant=True,
    )

    out_dir = train_cfg["out"] or os.path.join(
        "runs", "sidreasoner", f"{category}_stage2_activation"
    )
    run_sft(
        "activation",
        cfg,
        category,
        base_model,
        out_dir,
        tokenizer,
        num_new_tokens,
        train_data,
        val_sid,
        val_title2sid,
        val_sid2title,
    )


if __name__ == "__main__":
    main()
