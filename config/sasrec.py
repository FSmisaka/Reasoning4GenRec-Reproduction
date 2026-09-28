"""
SASRec 模型配置

Self-Attentive Sequential Recommendation
对应模型: src/models/sasrec.py
(MiniOneRec 原版实现的配置见 config/sasrec_mini.py)

数据集通过 DATASET 环境变量选择(见 data.categories),
如: DATASET=games make games-sasrec。
修改超参数直接编辑本文件,训练脚本不接受任何命令行超参。
"""

from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    "data": {
        "root": "data/raw/Amazon",
        "categories": {
            "games": "Video_Games",
            "office": "Office_Products",
        },
    },
    "model": {
        "max_seq_len": 10,
        "hidden": 128,
        "n_layers": 2,
        "n_heads": 2,
        "d_inner": 256,
        "dropout": 0.5,
        "init_std": 0.02,
        "layer_norm_eps": 1e-12,
        "neg_resample_rounds": 16,
    },
    "training": {
        "loss": "bce",
        "loss_choices": ["bce", "ce"],
        "optimizer": "adam",
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 500,
        "eval_every": 1,
        "patience": 10,
        "seed": 42,
        "out": None,
    },
    "evaluation": {
        "k_list": [5, 10],
        "batch_size": 512,
    },
}
