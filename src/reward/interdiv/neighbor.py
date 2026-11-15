"""In-batch Top-K 相似用户检索。

输入为当前 batch 去重后的 query 用户向量(L2 归一化), 通过一次矩阵乘
得到 BxB 余弦相似度矩阵, 再按行 top-K。GPU/CPU 自动选择, 纯 torch 原生
张量操作。

排除规则(经 exclude 掩码传入, 与检索解耦), 两条互补的判定:
- 精确: 缓存带 user_id 时(离线提取按行号 join 自 CSV, 见
  sidreasoner_user_embed._join_user_ids), 同 user_id 的样本一律排除;
- 启发: 历史物品集合 Jaccard >= 阈值(默认 0.8)视为同一用户 —— 覆盖
  缓存无 user_id 的情况, 以及同用户滑窗历史高度重叠的场景。
  (实测 Video_Games: 不同用户间 Jaccard p95=0.000, 阈值 0.8 不会误伤)
"""

import torch


def build_same_user_mask(
    history_sets: list, jaccard_threshold: float = 0.8, user_ids: list = None
) -> torch.Tensor:
    """(B, B) bool: True 表示两样本疑似同一用户(含自身)。

    Args:
        history_sets: 每个 query 的历史物品 frozenset。
        user_ids: 可选, 精确 user_id(空串/None 视为未知)。
    """
    b = len(history_sets)
    mask = torch.zeros(b, b, dtype=torch.bool)
    for i in range(b):
        mask[i, i] = True
    for i in range(b):
        uid_i = user_ids[i] if user_ids else ""
        for j in range(i + 1, b):
            same = False
            if uid_i and user_ids:
                same = uid_i == user_ids[j]
            if not same:
                a, c = history_sets[i], history_sets[j]
                if a and c:
                    inter = len(a & c)
                    if inter > 0 and inter / (len(a) + len(c) - inter) >= jaccard_threshold:
                        same = True
            if same:
                mask[i, j] = mask[j, i] = True
    return mask


def topk_neighbors(
    user_embeddings: torch.Tensor,
    k: int,
    exclude: torch.Tensor = None,
    device: str = "cpu",
):
    """每个 query 的 top-K 相似用户(余弦)。

    Args:
        user_embeddings: (B, D) float, 已 L2 归一化。
        k: 邻居数(不足时按实际可用数返回)。
        exclude: (B, B) bool True=不可作为邻居(自身/同用户)。
        device: 默认 cpu —— reward 在 verl driver 进程计算, BxB 的矩阵乘
            在 CPU 上是微秒级; 默认用 cuda 反而会在 colocated 的 GPU 0 上
            建 CUDA context(几百 MB), 挤占已调满的显存预算。

    Returns:
        neighbor_idx: (B, k') long(k' = min(k, B-1)), 不足的槽位填 -1;
        neighbor_sim: (B, k') float 对应相似度, 无效槽位为 NaN。
    """
    u = user_embeddings.to(device=device, dtype=torch.float32)
    b = u.shape[0]
    if b < 2:
        empty_idx = torch.full((b, 0), -1, dtype=torch.long)
        return empty_idx, torch.zeros((b, 0), dtype=torch.float32)

    k_eff = min(int(k), b - 1)
    sim = u @ u.T  # (B, B), 归一化后即余弦
    if exclude is not None:
        sim = sim.masked_fill(exclude.to(device), float("-inf"))
    else:
        sim.fill_diagonal_(float("-inf"))

    top_sim, top_idx = sim.topk(k_eff, dim=1)
    valid = torch.isfinite(top_sim)
    neighbor_idx = top_idx.masked_fill(~valid, -1).cpu()
    neighbor_sim = top_sim.masked_fill(~valid, float("nan")).cpu()
    return neighbor_idx, neighbor_sim
