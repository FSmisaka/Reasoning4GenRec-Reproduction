"""SIDReasoner Stage 3 跨 Query CoT 多样性奖励(verl batch 入口)。

启用方式(见 scripts/sidreasoner/rl.sh 的 INTER_DIV 分支):
- ``reward_model.reward_manager=batch``(verl 自带 BatchRewardManager,
  把整个逻辑 batch 的 B 个 query x G 个 rollout 一次性传入);
- ``custom_reward_function.path=<本文件>`` + ``.name=rule_base_reward``;
- ``custom_reward_function.reward_kwargs.*``:
    base_reward_type = steprule(默认, 复用 direct_recommendation_steprule)
                       | onerec_rollout_beam | onereason_div_rollout_beam
                       | ... (复用 sid_rewards 注册表)
    enable_inter_query_diversity / user_embedding_path / top_k_neighbors /
    lambda_init / lambda_final / decay_steps / gamma / sigma /
    same_user_jaccard / bleu_div_len / model_path / metrics_path ...

原 StepRule / Rollout-Beam reward 的逐行计算逻辑完全复用、不做修改;
本入口只在其上叠加多样性项, 关闭开关(enable=False)时逐行 score 与
原 reward 完全一致(无 batch 副作用)。

verl 通过 importlib 按路径加载本模块(driver 与 ray actor 进程均可能),
这里把仓库根目录挂到 sys.path, 保证 src.* 在任意进程可导入。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_ADDON = None
_BASE_COMPUTER = None
_BASE_KEY = None


def _bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    return str(x).lower() in ("1", "true", "yes", "on")


def _correctness_reward_type(kwargs: dict) -> str:
    """正确性 reward 类型: correctness_reward_type(新名) 优先,
    base_reward_type(旧名) 兼容。"""
    return str(
        kwargs.get("correctness_reward_type")
        or kwargs.get("base_reward_type")
        or "steprule"
    ).lower()


def _build_base_computer(kwargs: dict):
    """按正确性 reward 类型复用原有实现(逐行, 不改动)。

    可选值: steprule(SIDReasoner 论文原版)| onerec_rollout_beam
    (OneRec-Think)| onereason_div_rollout_beam(Rollout-Beam x R_div)|
    sid_rewards 注册表中的其他变体 —— 新正确性 reward 注册进
    src/reward/sid_rewards/ 即可, 无需改本入口。
    """
    base_type = _correctness_reward_type(kwargs)
    if base_type == "steprule":
        from src.reward.direct_recommendation_steprule import MyRewardComputer

        return MyRewardComputer(), base_type
    # sid_rewards 注册表(OneRec-Think Rollout-Beam 族)
    from src.reward.sid_rewards import build_sid_reward

    build_kwargs = dict(kwargs)
    build_kwargs["reward_type"] = base_type
    # 格式分路径与 rolloutbeam 入口同一套自动解析
    if (
        float(build_kwargs.get("format_reward_weight", 0.0) or 0.0) > 0
        and not build_kwargs.get("item_info_path")
    ):
        from src.reward.direct_recommendation_rolloutbeam import _default_item_info_path

        build_kwargs["item_info_path"] = _default_item_info_path()
    return build_sid_reward(build_kwargs), base_type


def _get_base_computer(kwargs: dict):
    global _BASE_COMPUTER, _BASE_KEY
    key = _correctness_reward_type(kwargs)
    if _BASE_COMPUTER is None or _BASE_KEY != key:
        _BASE_COMPUTER = _build_base_computer(kwargs)
        _BASE_KEY = key
    return _BASE_COMPUTER


def _get_addon(kwargs: dict):
    global _ADDON
    if _ADDON is None:
        from src.reward.interdiv import InterQueryDiversityAddon

        path = kwargs.get("user_embedding_path")
        if not path:
            raise ValueError(
                "enable_inter_query_diversity=True 需要 user_embedding_path"
                "(先运行 make <dataset>-sidreasoner-user-embed 生成离线缓存)。"
            )
        _ADDON = InterQueryDiversityAddon(
            user_embedding_path=kwargs["user_embedding_path"],
            top_k_neighbors=kwargs.get("top_k_neighbors", 8),
            same_user_jaccard=kwargs.get("same_user_jaccard", 0.8),
            bleu_div_len=kwargs.get("bleu_div_len", 1000),
            lambda_init=kwargs.get("lambda_init", 0.2),
            lambda_final=kwargs.get("lambda_final", 0.0),
            decay_steps=kwargs.get("decay_steps", 300),
            gamma=float(kwargs.get("gamma", 1.0)),
            sigma=kwargs.get("sigma", 0.5),
            model_path=kwargs.get("model_path"),
            metrics_path=kwargs.get("metrics_path"),
            log_every=int(kwargs.get("log_every", 1)),
            seed=kwargs.get("seed", 0),
            success_mode=str(kwargs.get("success_mode", "full")),
            reference_mode=str(kwargs.get("reference_mode", "full")),
            bleu_smoothing=str(kwargs.get("bleu_smoothing", "none")),
            td_norm=str(kwargs.get("td_norm", "none")),
            similarity=str(kwargs.get("similarity", "bleu")),
            ngram_n=int(kwargs.get("ngram_n", 1)),
            combine=str(kwargs.get("combine", "add")),
            judge_base_url=str(kwargs.get("judge_base_url", "http://localhost:8000/v1")),
            judge_model=str(kwargs.get("judge_model", "Qwen3-1.7B")),
            judge_concurrency=int(kwargs.get("judge_concurrency", 32)),
        )
    return _ADDON


def rule_base_reward(
    data_sources=None,
    solution_strs=None,
    ground_truths=None,
    extra_infos=None,
    **kwargs,
):
    """verl BatchRewardManager 兼容入口(返回 list[float] 或 list[dict])。

    list[dict] 时 BatchRewardManager 取 ``score`` 为 reward, 其余键
    进入 reward_extra_info(随 rollout dump 落盘)。默认返回 dict,
    便于离线分析每条 rollout 的 base/td/intrinsic 分解。
    """
    kwargs = dict(kwargs or {})
    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [None] * n

    computer, _ = _get_base_computer(kwargs)
    base_rewards = [
        float(
            computer.compute(
                data_source=data_sources[i],
                solution_str=solution_strs[i],
                ground_truth=ground_truths[i],
                extra_info=extra_infos[i],
            )
        )
        for i in range(n)
    ]

    if not _bool(kwargs.get("enable_inter_query_diversity"), default=True):
        # 显式关闭: 原样返回原始 reward(与未启用本入口时的行为一致)
        return [
            {"score": b, "base_reward": b, "success": 0, "td": 0.0, "intrinsic": 0.0}
            for b in base_rewards
        ]

    addon = _get_addon(kwargs)
    return addon.compute_batch(
        rows=[
            {
                "data_source": data_sources[i],
                "solution_str": solution_strs[i],
                "ground_truth": ground_truths[i],
                "extra_info": extra_infos[i],
            }
            for i in range(n)
        ],
        base_rewards=base_rewards,
    )
