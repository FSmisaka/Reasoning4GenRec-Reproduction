"""
SASRec MiniOneRec 模型配置

对应 MiniOneRec 项目的原版 SASRec 实现
对应模型: src/models/sasrec_mini.py

数据集通过 DATASET 环境变量选择(见 data.categories),
如: DATASET=games make games-sasrec-mini。
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
        "hidden_factor": 32,
        "seq_size": 10,
        "num_heads": 1,
        "dropout": 0.3,
        "init_std": 0.01,
    },
    "training": {
        "optimizer": "adam",
        "lr": 1e-3,
        "adam_eps": 1e-8,
        "l2_decay": 1e-5,
        "batch_size": 1024,
        "epochs": 500,
        "patience": 20,
        "seed": 1,
        "out": None,
    },
    "evaluation": {
        "k_list": [5, 10, 20],
        "batch_size": 1024,
    },
}
