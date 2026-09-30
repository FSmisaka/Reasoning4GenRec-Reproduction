"""
Caser 模型配置

Personalized Top-N Sequential Recommendation via Convolutional
Sequence Embedding (Tang & Wang, WSDM 2018)
对应模型: src/models/caser.py
原始实现: https://github.com/graytowne/caser_pytorch

卷积序列编码器(垂直/水平卷积核数量 nv/nh、隐维度 d、dropout、
激活函数 ac_conv/ac_fc)沿用原始实现的默认值:d=50, nv=4, nh=16,
drop=0.5, relu/relu。

打分为共享物品嵌入点积(GAMER 判别式基线协议, 与原始实现的
差异及原因见 src/models/caser.py 的类 docstring);
训练目标对齐 SIDReasoner(KDD'26)附录 A: 默认单目标 BCE
+ 3 个均匀负例(负例数与原始 Caser 实现一致), 其结果与论文
报告值接近; loss="ce"(全词表 CrossEntropy)会显著超出论文值,
仅供对照。L=10 使用完整历史窗口(与其他模型一致)。

数据集通过 DATASET 环境变量选择(见 data.categories),
如: DATASET=games make games-caser。
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
        "L": 10,
        "d": 50,
        "nv": 4,
        "nh": 16,
        "drop": 0.5,
        "ac_conv": "relu",
        "ac_fc": "relu",
    },
    "training": {
        "loss": "bce",
        "loss_choices": ["bce", "ce"],
        "neg_samples": 3,
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
