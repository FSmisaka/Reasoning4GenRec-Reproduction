import copy
import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_category(data_cfg):
    categories = data_cfg.get("categories", {})
    dataset = os.environ.get("DATASET")
    if dataset is None:
        raise RuntimeError(
            f"DATASET 环境变量未设置,可选: {sorted(categories)}"
        )
    if dataset not in categories:
        raise RuntimeError(
            f"未知 DATASET '{dataset}',可选: {sorted(categories)}"
        )
    return categories[dataset]


def run_config(cfg, category):
    config = copy.deepcopy(cfg)
    config["data"]["category"] = category
    return config


def save_run(out_dir, config, best_valid, test_metrics, paper_ref):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "config": config,
        "best_valid": best_valid,
        "test": test_metrics,
        "paper_reference": paper_ref,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[run] saved {out_dir}/metrics.json")


PAPER_REF = {
    ("Video_Games", "SASRec"): {
        "Recall@5": 0.0501, "NDCG@5": 0.0345,
        "Recall@10": 0.0723, "NDCG@10": 0.0416,
    },
    ("Video_Games", "TIGER"): {
        "Recall@5": 0.0489, "NDCG@5": 0.0300,
        "Recall@10": 0.0763, "NDCG@10": 0.0402,
    },
    ("Office_Products", "SASRec"): {
        "Recall@5": 0.1019, "NDCG@5": 0.0824,
        "Recall@10": 0.1167, "NDCG@10": 0.0871,
    },
    ("Office_Products", "TIGER"): {
        "Recall@5": 0.1270, "NDCG@5": 0.1037,
        "Recall@10": 0.1429, "NDCG@10": 0.1121,
    },
}


def print_comparison(model_name, category, test_metrics):
    ref = PAPER_REF.get((category, model_name))
    print(f"\n===== {model_name} on {category} (test) =====")
    print(f"{'metric':<12}{'repro':>10}{'paper':>10}")
    for k in ["Recall@5", "NDCG@5", "Recall@10", "NDCG@10"]:
        paper = f"{ref[k]:>10.4f}" if ref is not None else f"{'n/a':>10}"
        print(f"{k:<12}{test_metrics[k]:>10.4f}{paper}")
