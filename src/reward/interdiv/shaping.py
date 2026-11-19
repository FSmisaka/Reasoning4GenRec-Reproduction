"""DIVER 风格 Reward Shaping(跨 Query 改造版)。

按任务设计(源自 DIVER 论文 Sec 3.4-3.5 的 Potential-based Shaping,
比较集合替换为 in-batch 相似用户后的改造):

    R'_{q,i} = R_{q,i} + lambda(t) * I_{q,i} * clip( gamma^T_{q,i} * d(o_{q,i}), 0, sigma )

- R_{q,i}: 原始推荐 reward, 原样保留;
- I_{q,i}: 推荐成功指示(SID 三级全匹配), 条件塑形 —— 只奖励成功 rollout;
- lambda(t): 线性衰减(lambda_init -> lambda_final, 前 decay_steps 步);
- gamma^T: 折扣(默认 gamma=1.0 时恒为 1, 不需要 token 数);
- sigma: 裁剪上界(Balanced Shaping, 防奖励黑客)。

注意: 跨 Query 设置下 Reference 依赖 batch 内其他用户, 原 DIVER 的
最优策略不变性定理不再自动成立, 本模块视为待实验验证的 reward 设计。
"""

import numpy as np


def lambda_at(step: int, lambda_init: float, lambda_final: float, decay_steps: int) -> float:
    """线性插值衰减: step >= decay_steps 后停在 lambda_final。"""
    if decay_steps <= 0:
        return float(lambda_final)
    t = min(max(step, 0), decay_steps)
    return float(lambda_init + (lambda_final - lambda_init) * t / decay_steps)


def shape_rewards(
    base_rewards,
    success_mask,
    td_values,
    token_counts=None,
    gamma: float = 1.0,
    sigma: float = 0.5,
    lam: float = 0.0,
    combine: str = "add",
):
    """逐 rollout 计算 intrinsic 与 augmented reward(纯 numpy)。

    Args:
        base_rewards: (N,) 原始推荐 reward(正确性 reward), 原样保留。
        success_mask: (N,) bool, 推荐成功指示 I。
        td_values: (N,) float, 跨用户 TD d(o)。
        token_counts: (N,) 可选, rollout token 数 T(gamma<1 时用于 gamma^T)。
        gamma: 折扣因子(默认 1.0)。
        sigma: intrinsic 裁剪上界。
        lam: 当前步多样性权重 lambda(t)。
        combine: 正确性 x 多样性的合并方式:
            - "add"(默认): R' = R + intrinsic(DIVER 塑形);
            - "mul":       R' = R * (1 + intrinsic) —— 多样性按比例放大
              正确性 reward(R=0 的 rollout 不获得任何加成; 与
              rl-beam-div 的乘法思想同源, 但保留 DIVER 的 lambda/sigma/
              条件塑形语义)。

    Returns:
        (intrinsic: (N,) float, augmented: (N,) float)
    """
    base = np.asarray(base_rewards, dtype=np.float64)
    td = np.asarray(td_values, dtype=np.float64)
    success = np.asarray(success_mask, dtype=bool)

    if gamma == 1.0 or token_counts is None:
        discounted = td
    else:
        counts = np.asarray(token_counts, dtype=np.float64)
        discounted = np.power(float(gamma), counts) * td

    clipped = np.clip(discounted, 0.0, float(sigma))
    intrinsic = float(lam) * success.astype(np.float64) * clipped
    if combine == "mul":
        return intrinsic, base * (1.0 + intrinsic)
    if combine == "add":
        return intrinsic, base + intrinsic
    raise ValueError(f"未知 combine: {combine!r}(add | mul)")


def rank_normalize(values) -> "np.ndarray":
    """把一组值按秩映射到 [0, 1] 均布(并列取平均秩, n<2 返回全 0)。

    用于 td_norm="rank": GRPO 只关心组内相对排序, 而 TD 的组内绝对差
    通常只有 ~0.01; 秩归一化让组内区分度由 lambda*sigma 直接控制。
    """
    vals = np.asarray(values, dtype=np.float64)
    n = vals.size
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    ranks = np.empty(n, dtype=np.float64)
    order = np.argsort(vals, kind="stable")
    ranks[order] = np.arange(n, dtype=np.float64)
    uniq, inv = np.unique(vals, return_inverse=True)
    for u_i in range(uniq.size):
        idx = np.where(inv == u_i)[0]
        if idx.size > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks / (n - 1)
