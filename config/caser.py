"""
Caser 模型配置

Personalized Top-N Sequential Recommendation via Convolutional
Sequence Embedding (Tang & Wang, WSDM 2018)
对应模型: src/models/caser.py
原始实现: https://github.com/graytowne/caser_pytorch

卷积序列编码器(垂直/水平卷积核数量 nv/nh、隐维度 d、dropout、
激活函数 ac_conv/ac_fc)沿用原始实现的默认值:d=50, nv=4, nh=16,
drop=0.5, relu/relu。

打分与训练协议对齐 GAMER(SeqRec.modules.model_base.seq_model.SeqModel)
判别式基线的统一设定, 与原始 graytowne 实现的差异及原因见
src/models/caser.py 的类 docstring:
- 无用户嵌入(评估用户过半未在训练集出现, 个性化无法泛化);
- 共享物品嵌入打分 + 全词表 CrossEntropy(逐物品 W2/b2 自由参数 +
  负采样 BCE 在本数据集会坍缩为反流行度排序);
- L=10 使用完整历史窗口(与其他模型一致, 数据集历史截断为 10)。

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
