"""Stage 3 (GRPO) 可插拔 SID reward 的 verl 入口。

默认配置 = OneRec-Think Rollout-Beam reward(arXiv:2510.11639, Eq. 6-7)。
其他注册的 reward 变体(见 src/reward/sid_rewards/)通过
config/sidreasoner.py 的 rl.reward_kwargs.reward_type 切换, 无需改任何
trainer 代码::

    # rl.sh 据此下发(见 config/sidreasoner.py 的 rl 段):
    custom_reward_function.path="<本文件绝对路径>"
    custom_reward_function.name="rule_base_reward"
    custom_reward_function.reward_kwargs.reward_type="onerec_rollout_beam"
    custom_reward_function.reward_kwargs.num_sid_levels=3

本文件为增量入口: 原StepRule reward(src/reward/direct_recommendation_steprule.py,
rl.reward=steprule 时的默认)不受影响。

verl 通过 importlib 按文件路径加载本模块(驱动进程与 ray actor 进程都有
可能), 因此这里先把本目录挂到 sys.path, 保证 sid_rewards 包在任意进程
中都能导入。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sid_rewards import build_sid_reward  # noqa: E402

# info 文件名与 direct_recommendation_steprule.py 保持一致
_CATEGORY_FILES = {
    "Video_Games": "Video_Games_5_2016-10-2018-11.txt",
    "Office_Products": "Office_Products_5_2016-10-2018-11.txt",
    "Industrial_and_Scientific": "Industrial_and_Scientific_5_2016-10-2018-11.txt",
}


def _default_item_info_path():
    """按 DATASET 环境变量 + config 解析当前数据集的 info 文件路径。

    仅在开启格式分(format_reward_weight > 0)且未显式给
    reward_kwargs.item_info_path 时才会被调用。
    """
    if "." not in sys.path:
        sys.path.insert(0, ".")
    from config import load_config

    cfg = load_config("sidreasoner")
    category = os.environ.get("DATASET")
    if category is None or category not in cfg["data"]["categories"]:
        raise RuntimeError(
            f"DATASET 环境变量未设置或无效: {category!r}, "
            f"可选: {sorted(cfg['data']['categories'])}"
        )
    fname = _CATEGORY_FILES.get(cfg["data"]["categories"][category])
    if fname is None:
        raise RuntimeError(f"未知 category: {cfg['data']['categories'][category]}")
    return os.path.join(cfg["data"]["root"], "info", fname)


def rule_base_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl 兼容的 reward 包装: 转发到 sid_rewards 注册表选中的变体。

    额外关键字参数(verl 从 custom_reward_function.reward_kwargs 合并而来)
    原样传给所选 reward 类; 实例按 (reward_type, 配置) 懒构建一次。
    """
    kwargs = dict(kwargs or {})
    # 开格式分但没给 item_info_path 时, 按当前 DATASET 自动解析(与
    # direct_recommendation_steprule.py 同一套路径规则)
    if float(kwargs.get("format_reward_weight", 0.0) or 0.0) > 0 and not kwargs.get("item_info_path"):
        kwargs["item_info_path"] = _default_item_info_path()

    computer = build_sid_reward(kwargs)
    return computer.compute(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
