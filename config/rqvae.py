"""
RQ-VAE 自建语义 ID(SID)配置

对应模块: src/models/rqvae.py, src/train/rqvae.py,
          src/data/rqvae_embed.py, src/data/rqvae_apply.py

官方 SID 由上游 RQ-VAE 离线计算并随数据发布(3 层 x 每层 256 code,
即 <a_N><b_N><c_N>); 本配置用于在 item.json 文本上独立训练一个
RQ-VAE, 得到一份自己的 SID 并级联替换 data/raw/Amazon 下的 4 处
SID 文件(index.json / CSV 的两列 / info txt / RL parquet)。

流水线三步(Makefile 目标 <dataset>-rqvae-<step>):
1. embed:  item.json 文本 -> BGE 向量(runs/rqvae/<Category>/embeddings.npy)
2. train:  训练 RQ-VAE -> 新 SID 表(runs/rqvae/<Category>/index.own.json)
3. apply:  备份官方文件到 data/raw/Amazon_official_backup/ 后级联替换

数据集通过 DATASET 环境变量选择(见 data.categories),
如: DATASET=games make games-rqvae。
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
        "suffix": "_5_2016-10-2018-11",
    },
    # 文本 -> 向量。BGE 系列对 passage 不需要 instruction 前缀,
    # 采用 CLS pooling + L2 归一化(与官方用法一致)。
    # text_fields 控制参与拼接的字段: 实测 description 的大量模板化
    # 文案会淹没 title 信号, 导致量化碰撞(Skylanders 系列 24 连撞),
    # 因此默认只用 title+brand。
    "embed": {
        "model": "BAAI/bge-base-en-v1.5",
        "text_fields": ["title", "brand"],
        "batch_size": 64,
        "max_length": 512,
    },
    # 结构与消费侧对齐: num_levels x codebook_size 必须保持 3 x 256,
    # 否则 <a_N>/<b_N>/<c_N> 词表(768 个 token)与 trie/评估代码失配。
    "model": {
        "input_dim": 768,
        "hidden_dims": [512],
        "code_dim": 256,
        "codebook_size": 256,
        "num_levels": 3,
        "ema_decay": 0.995,
        "commitment_beta": 0.1,
        "kmeans_init": True,
        "kmeans_iters": 50,
    },
    "training": {
        "epochs": 6000,
        "batch_size": 256,
        "lr": 1e-3,
        "seed": 42,
        "log_every": 300,
        "out": None,  # 默认 runs/rqvae/<Category>
    },
}
