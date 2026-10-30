"""可插拔的 Stage 3 (GRPO) reward 包。

把 reward 计算与 verl trainer 解耦: reward 变体只通过
config/sidreasoner.py 的 rl.reward + rl.reward_kwargs 选择,
无需改动 trainer / rollout 代码:

- ``base.py``        : SID 解析工具 + ``BaseSidReward`` 基类。
- ``registry.py``    : ``@register_sid_reward("name")`` 注册器与工厂。
- ``rollout_beam.py``: OneRec-Think Rollout-Beam reward
  (arXiv:2510.11639, Eq. 6-7)。
- 新变体             : 继承 ``BaseSidReward``, 加装饰器注册,
  然后 rl.reward_kwargs.reward_type=your_name 切换。

包为纯 Python(无 verl/torch 依赖), 内部一律相对导入,
既能以 ``src.reward.sid_rewards`` 导入(仓库根在 sys.path 时),
也能被入口文件 src/reward/direct_recommendation_rolloutbeam.py
在任意进程(含 ray actor)中按目录加载。
"""

from .base import (
    BEAM_SEP,
    BaseSidReward,
    construct_prefix_allowed_hashmap,
    extract_sid_tokens,
    extract_solution,
    parse_beam_candidates,
)
from .registry import (
    build_sid_reward,
    get_sid_reward_cls,
    register_sid_reward,
)

# 导入 reward 实现以完成注册
from . import rollout_beam  # noqa: F401

__all__ = [
    "BEAM_SEP",
    "BaseSidReward",
    "construct_prefix_allowed_hashmap",
    "extract_sid_tokens",
    "extract_solution",
    "parse_beam_candidates",
    "build_sid_reward",
    "get_sid_reward_cls",
    "register_sid_reward",
]
