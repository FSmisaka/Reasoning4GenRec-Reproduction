"""name -> class 的 reward 注册表。

新增 reward 变体(放本包内, 如 my_reward.py)::

    from .base import BaseSidReward
    from .registry import register_sid_reward

    @register_sid_reward("my_reward")
    class MyReward(BaseSidReward):
        def compute(self, data_source, solution_str, ground_truth, extra_info):
            ...

然后在 config/sidreasoner.py 里::

    "rl": {
        "reward": "rollout_beam",
        "reward_kwargs": {"reward_type": "my_reward", ...},
    }

同一 (reward_type, 配置) 只构建一次实例(构建 SID 前缀表等较贵)。
"""

import json
import threading

_REWARD_REGISTRY: dict = {}

# 每个 (reward_type, 配置) 一个懒构建单例
_BUILT_REWARDS: dict = {}
_REGISTRY_LOCK = threading.Lock()


def register_sid_reward(name: str):
    """类装饰器: 把 BaseSidReward 子类注册到 ``name``。"""

    def decorator(cls):
        key = str(name).lower()
        if key in _REWARD_REGISTRY:
            raise ValueError(f"SID reward '{name}' 已注册。")
        _REWARD_REGISTRY[key] = cls
        return cls

    return decorator


def get_sid_reward_cls(name: str):
    key = str(name).lower()
    if key not in _REWARD_REGISTRY:
        available = ", ".join(sorted(_REWARD_REGISTRY)) or "<none>"
        raise KeyError(
            f"未知的 reward_type '{name}', 可选: {available}。"
            "自定义 reward 请继承 BaseSidReward 并用 @register_sid_reward 注册。"
        )
    return _REWARD_REGISTRY[key]


def build_sid_reward(kwargs: dict):
    """按 ``kwargs`` 构建(或返回缓存的) reward 实例。

    ``reward_type`` 选择注册的类(默认 onerec_rollout_beam),
    其余键原样传给类构造函数。
    """
    kwargs = dict(kwargs or {})
    reward_type = str(kwargs.pop("reward_type", "onerec_rollout_beam"))
    cache_key = reward_type + "::" + json.dumps(kwargs, sort_keys=True, default=str)
    with _REGISTRY_LOCK:
        if cache_key not in _BUILT_REWARDS:
            cls = get_sid_reward_cls(reward_type)
            _BUILT_REWARDS[cache_key] = cls(**kwargs)
        return _BUILT_REWARDS[cache_key]
