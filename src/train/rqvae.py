"""
RQ-VAE 第 2 步: 训练 + 量化导出自建 SID

启动: DATASET=games make games-rqvae-train
      (等价: DATASET=games .venv/bin/python -m src.train.rqvae)
前置: 先跑 src.data.rqvae_embed 生成 embeddings.npy
输出: runs/rqvae/<Category>/checkpoint.pt, index.own.json, report.json

index.own.json 的格式与官方 index/<Category>.index.json 完全一致
({"item_id": ["<a_N>", "<b_N>", "<c_N>"], ...}), 但不直接覆盖官方
文件 —— 级联替换由 src.data.rqvae_apply 负责(带备份)。
"""

import json
import os
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import load_config
from src.models.rqvae import RQVAE
from src.data.rqvae_embed import output_dir
from src.train.common import pick_device, resolve_category, set_seed

_LEVEL_LETTERS = "abc"


def codes_to_tokens(codes):
    return [f"<{_LEVEL_LETTERS[l]}_{int(c)}>" for l, c in enumerate(codes)]


def level_usage(codes, level):
    return len(set(codes[:, level].tolist())) if len(codes) else 0


def compare_with_official(new_sid, old_path):
    """与官方 SID 对比: 独立训练预期接近随机, 仅作参照。

    new_sid: item -> code 元组; 官方表为 token 列表, 用正则抽 code。
    """
    if not os.path.exists(old_path):
        return None
    token_re = re.compile(r"<[abc]_(\d+)>")
    old_sid = {
        int(k): tuple(int(m) for m in token_re.findall("".join(v)))
        for k, v in json.load(open(old_path)).items()
    }
    common = sorted(set(new_sid) & set(old_sid))
    stats = {"n_common": len(common), "exact_match": 0, "per_level_match": [0] * 3}
    for item in common:
        a, b = new_sid[item], old_sid[item]
        stats["exact_match"] += int(a == b)
        for l in range(len(a)):
            stats["per_level_match"][l] += int(a[l] == b[l])
    stats["exact_match_rate"] = stats["exact_match"] / max(len(common), 1)
    stats["per_level_match_rate"] = [
        m / max(len(common), 1) for m in stats["per_level_match"]
    ]
    return stats


def main(cfg=None):
    cfg = cfg or load_config("rqvae")
    data_cfg = cfg["data"]
    category = resolve_category(data_cfg)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    out_dir = train_cfg.get("out") or output_dir(category)
    os.makedirs(out_dir, exist_ok=True)
    emb_path = os.path.join(out_dir, "embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"未找到 {emb_path}, 先运行: DATASET=<dataset> "
            "make <dataset>-rqvae-embed"
        )

    set_seed(train_cfg["seed"])
    device = pick_device()
    X = torch.from_numpy(np.load(emb_path)).float().to(device)

    model = RQVAE(
        input_dim=model_cfg["input_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        code_dim=model_cfg["code_dim"],
        codebook_size=model_cfg["codebook_size"],
        num_levels=model_cfg["num_levels"],
        ema_decay=model_cfg["ema_decay"],
        commitment_beta=model_cfg["commitment_beta"],
        kmeans_init=model_cfg["kmeans_init"],
        kmeans_iters=model_cfg["kmeans_iters"],
    ).to(device)
    if model_cfg["kmeans_init"]:
        model.init_codebooks(X)
        print("[train] codebook 已用逐层 k-means 初始化")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=train_cfg["lr"]
    )
    loader = DataLoader(
        TensorDataset(X),
        batch_size=min(train_cfg["batch_size"], len(X)),
        shuffle=True,
    )

    epochs = train_cfg["epochs"]
    log_every = train_cfg["log_every"]
    model.train()
    for epoch in range(1, epochs + 1):
        for (batch,) in loader:
            loss, info = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % log_every == 0 or epoch == epochs:
            with torch.no_grad():
                codes = model.quantize(X)
                recon = model.decoder(
                    sum(
                        torch.nn.functional.embedding(
                            codes[:, l], model.levels[l].embed
                        )
                        for l in range(model.num_levels)
                    )
                )
                cos = torch.nn.functional.cosine_similarity(
                    recon, X, dim=1
                ).mean().item()
            usage = [
                level_usage(codes, l) for l in range(model.num_levels)
            ]
            print(
                f"[train] epoch {epoch:>5} loss {loss.item():.5f} "
                f"recon {info['recon'].item():.5f} "
                f"commit {info['commit'].item():.5f} cos {cos:.4f} "
                f"usage/256 {usage}"
            )

    model.eval()
    with torch.no_grad():
        codes = model.quantize(X)
        recon = model.decoder(
            sum(
                torch.nn.functional.embedding(
                    codes[:, l], model.levels[l].embed
                )
                for l in range(model.num_levels)
            )
        )
        cos = (
            torch.nn.functional.cosine_similarity(recon, X, dim=1)
            .mean()
            .item()
        )
        codes = codes.cpu().numpy()

    with open(os.path.join(data_cfg["root"], "index", f"{category}.item.json")) as f:
        item_ids = sorted(json.load(f), key=int)
    assert len(item_ids) == len(codes), "item.json 与 embeddings 数量不一致"

    item2tokens = {
        k: codes_to_tokens(codes[i]) for i, k in enumerate(item_ids)
    }
    index_path = os.path.join(out_dir, "index.own.json")
    with open(index_path, "w") as f:
        f.write("{\n")
        entries = sorted(item2tokens.items(), key=lambda kv: int(kv[0]))
        for i, (k, v) in enumerate(entries):
            comma = "," if i < len(entries) - 1 else ""
            f.write(f"{json.dumps(k)}: {json.dumps(v)}{comma}\n")
        f.write("}\n")

    sids = [tuple(row) for row in codes]
    sid_counts = Counter(sids)
    report = {
        "category": category,
        "n_items": len(item_ids),
        "n_unique_sids": len(sid_counts),
        "n_collisions": len(item_ids) - len(sid_counts),
        "level_usage": {
            f"level_{l}": level_usage(codes, l)
            for l in range(model.num_levels)
        },
        "recon_cosine": cos,
        "config": cfg,
        "vs_official": compare_with_official(
            {int(k): tuple(int(c) for c in codes[i])
             for i, k in enumerate(item_ids)},
            os.path.join(data_cfg["root"], "index", f"{category}.index.json"),
        ),
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "item_ids": item_ids},
        os.path.join(out_dir, "checkpoint.pt"),
    )
    print(
        f"[train] {category}: unique SID {report['n_unique_sids']}"
        f"/{report['n_items']} (collision {report['n_collisions']}), "
        f"usage/256 {report['level_usage']} -> {index_path}"
    )


if __name__ == "__main__":
    main()
