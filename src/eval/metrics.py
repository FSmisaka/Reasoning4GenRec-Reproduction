import math


def rank_metrics(ranks, k_list=(5, 10)):
    metrics = {}
    n = len(ranks)
    for k in k_list:
        hits = [r for r in ranks if 0 <= r < k]
        recall = len(hits) / n
        ndcg = sum(1.0 / math.log2(r + 2) for r in hits) / n
        metrics[f"Recall@{k}"] = recall
        metrics[f"NDCG@{k}"] = ndcg
    return metrics


def rank_of(target, ranked_items):
    for i, item in enumerate(ranked_items):
        if item == target:
            return i
    return -1
