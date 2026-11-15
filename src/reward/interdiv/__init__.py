"""Cross-Query CoT Diversity Reward 包(DIVER 改造版, 用于 SIDReasoner Stage 3)。

模块划分(相互解耦, 便于单独替换):
- bleu.py       DIVER 官方口径的 BLEU 相似度(无 nltk 依赖)
- cot_utils.py  CoT 提取与"推荐成功"判定(SID 三级全匹配)
- user_store.py 离线 User Embedding 缓存查表
- neighbor.py   In-batch Top-K 相似用户检索(排除自身/同用户)
- shaping.py    lambda 衰减 + 条件塑形 + 折扣裁剪
- reward.py     InterQueryDiversityAddon 编排器
"""

from .bleu import DiverBleu  # noqa: F401
from .reward import InterQueryDiversityAddon  # noqa: F401
from .user_store import UserEmbeddingStore  # noqa: F401

__all__ = ["DiverBleu", "InterQueryDiversityAddon", "UserEmbeddingStore"]
