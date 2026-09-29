"""
GRU4Rec 模型配置

Session-based Recommendation with Recurrent Neural Networks
(Hidasi and Karatzoglou, RecSys 2018; 第一版 ICLR 2016)
对应模型: src/models/gru4rec.py
参考实现: GAMER (https://github.com/wzf2000/GAMER,
SeqRec/models/discriminative/GRU4Rec), 其实现参考 RecBole

CONFIG["model"] 与 GAMER 的 GRU4RecConfig 默认值一致:
- embedding_size: 64
- hidden_size:    128
- n_layers:       1
- dropout:        0.3

打分与训练协议遵循 GAMER SeqModel 判别式基线的统一设定
(共享物品嵌入打分 + 全词表 CrossEntropy, 无用户嵌入),
说明见 src/models/gru4rec.py 的类 docstring。

数据集通过 DATASET 环境变量选择(见 data.categories),
如: DATASET=games make games-gru4rec。
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
        "embedding_size": 64,
        "hidden_size": 128,
        "n_layers": 1,
        "dropout": 0.3,
    },
    "training": {
        "lr": 1e-3,
        "l2": 1e-6,
        "batch_size": 512,
        "epochs": 200,
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
