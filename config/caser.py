"""
Caser 模型配置

Personalized Top-N Sequential Recommendation via Convolutional
Sequence Embedding (Tang & Wang, WSDM 2018)
对应模型: src/models/caser.py
原始实现: https://github.com/graytowne/caser_pytorch

CONFIG["model"] 与原始仓库 train_caser.py 的默认超参数一致:
- L:   序列长度(原始默认 5)
- T:   每个训练窗口的目标数(原始默认 3;本仓库数据每行只有单个
       目标物品,故默认 1,多目标窗口由滑动窗口内部产生)
- d:   隐向量维度(原始默认 50)
- nv:  垂直卷积核数量(原始默认 4)
- nh:  水平卷积核数量(原始默认 16)
- drop/ac_conv/ac_fc: dropout 与激活函数(原始默认 0.5/relu/relu)

训练部分同样沿用原始默认: lr=1e-3, l2(weight_decay)=1e-6,
batch_size=512, neg_samples=3(每个目标采 3 个负样本);
损失为原始的 sigmoid 二元交叉熵,负样本从用户未见过的物品中采样。

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
        },
    },
    "model": {
        "L": 5,
        "T": 1,
        "d": 50,
        "nv": 4,
        "nh": 16,
        "drop": 0.5,
        "ac_conv": "relu",
        "ac_fc": "relu",
    },
    "training": {
        "lr": 1e-3,
        "l2": 1e-6,
        "batch_size": 512,
        "neg_samples": 3,
        "neg_resample_rounds": 16,
        "epochs": 200,
        "eval_every": 2,
        "patience": 10,
        "seed": 42,
        "out": None,
    },
    "evaluation": {
        "k_list": [5, 10],
        "batch_size": 512,
    },
}
