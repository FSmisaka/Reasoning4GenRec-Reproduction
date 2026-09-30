"""
SASRec 模型配置

Self-Attentive Sequential Recommendation
对应模型: src/models/sasrec.py
(MiniOneRec 原版实现的配置见 config/sasrec_mini.py, 其结果与
SIDReasoner 报告值最接近)

训练目标对齐 SIDReasoner(KDD'26)附录 A 的基线协议: 默认
loss="bce1"(单目标 BCE + 1 个均匀负例), 容量对齐其基线规模
(hidden=32, 1 层, 1 头); loss="bce" 为原版 SASRec 的全位置
BCE, loss="ce" 为全词表 CrossEntropy —— 后两者在本数据上均会
显著超出论文报告值, 仅供对照。

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
            "industrial": "Industrial_and_Scientific",
        },
    },
    "model": {
        "max_seq_len": 10,
        "hidden": 32,
        "n_layers": 1,
        "n_heads": 1,
        "d_inner": 256,
        "dropout": 0.5,
        "init_std": 0.02,
        "layer_norm_eps": 1e-12,
        "neg_resample_rounds": 16,
    },
    "training": {
        "loss": "bce1",
        "loss_choices": ["bce1", "bce", "ce"],
        "neg_samples": 1,
        "optimizer": "adam",
        "lr": 1e-3,
        "l2": 1e-6,
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
