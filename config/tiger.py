"""
TIGER 模型配置

Recommender Systems with Generative Retrieval
对应模型: src/models/tiger.py

包含:
- CONFIG["sid"]:        语义 ID(Semantic ID)表结构(src/data/sid.py)
- CONFIG["model"]:      T5 结构超参数(src/models/tiger/model.py 的 build_tiger)
- CONFIG["training"]:   训练超参数
- CONFIG["generation"]: 束搜索生成超参数(beam_search_items / evaluate_tiger)

数据集通过 DATASET 环境变量选择(见 data.categories),
如: DATASET=games make games-tiger。
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
    "sid": {
        "codebook_size": 256,
        "num_levels": 3,
        "pad_id": 0,
        "eos_id": 1,
        "special_tokens": 2,
    },
    "model": {
        "d_model": 128,
        "d_ff": 1024,
        "num_layers": 4,
        "num_heads": 6,
        "dropout": 0.1,
        "feed_forward_proj": "relu",
    },
    "training": {
        "max_items": 10,
        "optimizer": "adam",
        "optimizer_choices": ["adam", "adafactor"],
        "scheduler": "invsq",
        "scheduler_choices": ["constant", "invsq"],
        "warmup_steps": 10000,
        "lr": 5e-4,
        "batch_size": 256,
        "epochs": 200,
        "eval_every": 1,
        "eval_subset": 2000,
        "patience": 10,
        "seed": 42,
        "out": None,
    },
    "generation": {
        "num_beams": 20,
        "top_k": 10,
        "batch_size": 64,
    },
    "evaluation": {
        "k_list": [5, 10],
    },
}
