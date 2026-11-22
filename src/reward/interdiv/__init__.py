"""Cross-Query CoT Diversity Reward 包(DIVER 改造版, 用于 SIDReasoner Stage 3)。

模块划分(相互解耦, 便于单独替换):
- bleu.py       DIVER 官方口径的 BLEU 相似度(无 nltk 依赖)
- llmjudge.py   LLM-as-a-Judge 语义相似度(vLLM OpenAI 兼容, 异步批量)
- niche.py      Niche(小众物品)多样性 scorer(c1 信息提取/c2 句风反差/c3 总结句)
- cot_utils.py  CoT 提取与"推荐成功"判定(SID 三级全匹配)
- user_store.py 离线 User Embedding 缓存查表
- neighbor.py   In-batch Top-K 相似用户检索(排除自身/同用户)
- shaping.py    lambda 衰减 + 条件塑形 + 折扣裁剪
- reward.py     InterQueryDiversityAddon 编排器
"""

from .bleu import DiverBleu  # noqa: F401
from .llmjudge import HistoryLookup, LLMJudge, render_block  # noqa: F401
from .niche import NicheDiversityScorer  # noqa: F401
from .reward import InterQueryDiversityAddon  # noqa: F401
from .similarity import NgramJaccard  # noqa: F401
from .user_store import UserEmbeddingStore  # noqa: F401

__all__ = [
    "DiverBleu",
    "HistoryLookup",
    "InterQueryDiversityAddon",
    "LLMJudge",
    "NicheDiversityScorer",
    "NgramJaccard",
    "UserEmbeddingStore",
    "render_block",
]
