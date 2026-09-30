"""
SIDReasoner Stage 1: 增强式 SID-语言对齐 SFT

迁移自 SIDReasoner 官方仓库的 sft_Qwen3.py + sft_Qwen3_enrich.sh。
多任务混合训练(序列推荐 / SID 翻译 / 跨形式推荐 / LLM 增强语料 /
通用推理数据), 对应论文 3.2 节 Enriched SID-Language Alignment。

启动(4 卡示例, 数据集经 DATASET 环境变量选择):
  DATASET=games .venv/bin/python -m torch.distributed.run \
      --nproc_per_node 4 --master_port 12340 -m src.train.sidreasoner_sft
或 scripts/sidreasoner/sft.sh / make games-sidreasoner-sft
"""

import os

from config import load_config
from src.data.sidreasoner_data import (
    FusionSeqRecDataset,
    GeneralSFTReasonDataset,
    SFTData,
    SidItemFeatDataset,
    SidSFTDataset,
    SidTextInterleaveSequenceDataset,
    SidTextInterleaveDataset_v2,
    TitleHistory2SidSFTDataset,
)
from src.models.sidreasoner import (
    category_phrase,
    load_sidreasoner_tokenizer,
)
from src.train.common import resolve_category
from src.train.sidreasoner_common import run_sft
from torch.utils.data import ConcatDataset


def main(cfg=None):
    cfg = cfg or load_config("sidreasoner")
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    train_cfg = cfg["training"]["sft"]
    category = resolve_category(data_cfg)
    phrase = category_phrase(category)

    root = data_cfg["root"]
    fname = f"{category}_5_2016-10-2018-11.csv"
    paths = {
        "train_file": os.path.join(root, "train", fname),
        "eval_file": os.path.join(root, "valid", fname),
        "index_file": os.path.join(root, "index", f"{category}.index.json"),
        "item_meta": os.path.join(root, "index", f"{category}.item.json"),
        "enhanced": os.path.join(
            root, "index", f"{category}.item_enhanced_v2.json"
        ),
        "narrative": os.path.join(
            root, "index", f"{category}.integrated_narrative.csv"
        ),
        "general_reasoning": os.path.join(
            root, "general", "sampled_data.arrow"
        ),
    }

    # 先加载并扩展 SID token, 数据集与模型共用同一 tokenizer,
    # 保证 <a_N>/<b_N>/<c_N> 均为单个新词表条目
    tokenizer, num_new_tokens = load_sidreasoner_tokenizer(
        model_cfg["base_model"], paths["index_file"]
    )

    max_len = train_cfg["cutoff_len"]
    seed = train_cfg["seed"]
    sample = train_cfg.get("sample", -1)
    mask_assistant = train_cfg.get("mask_assistant", True)

    train_datasets = [
        SidSFTDataset(
            train_file=paths["train_file"], tokenizer=tokenizer,
            max_len=max_len, sample=sample, seed=seed, category=phrase,
            mask_assistant=mask_assistant,
        ),
        SidItemFeatDataset(
            item_file=paths["item_meta"], index_file=paths["index_file"],
            tokenizer=tokenizer, max_len=max_len, sample=sample,
            seed=seed, category=phrase, mask_assistant=mask_assistant,
        ),
        FusionSeqRecDataset(
            train_file=paths["train_file"], item_file=paths["item_meta"],
            index_file=paths["index_file"], tokenizer=tokenizer,
            max_len=max_len, sample=sample, seed=seed, category=phrase,
            mask_assistant=mask_assistant,
        ),
        SFTData(
            train_file=paths["train_file"], tokenizer=tokenizer,
            max_len=max_len, sample=sample, seed=seed, category=phrase,
            mask_assistant=mask_assistant,
        ),
        TitleHistory2SidSFTDataset(
            train_file=paths["train_file"], item_file=paths["item_meta"],
            index_file=paths["index_file"], tokenizer=tokenizer,
            max_len=max_len, sample=sample, seed=seed, category=phrase,
            mask_assistant=mask_assistant,
        ),
        SidTextInterleaveDataset_v2(
            json_file=paths["enhanced"], tokenizer=tokenizer,
            max_len=max_len, sample=sample, seed=seed,
        ),
        SidTextInterleaveSequenceDataset(
            csv_file=paths["narrative"], tokenizer=tokenizer,
            max_len=max_len, sample=sample, seed=seed,
        ),
        GeneralSFTReasonDataset(
            train_file=paths["general_reasoning"], tokenizer=tokenizer,
            max_len=train_cfg.get("general_max_len", 3072),
            sample=train_cfg.get("general_sample", 60000), seed=seed,
        ),
    ]
    train_data = ConcatDataset(train_datasets)

    val_sid = SidSFTDataset(
        train_file=paths["eval_file"], tokenizer=tokenizer, max_len=max_len,
        sample=sample, seed=seed, category=phrase, test=False,
        mask_assistant=True,
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
        "runs", "sidreasoner", f"{category}_stage1_sft"
    )
    run_sft(
        "sft",
        cfg,
        category,
        model_cfg["base_model"],
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
