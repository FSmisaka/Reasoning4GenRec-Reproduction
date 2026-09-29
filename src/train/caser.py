import os
import random
import time

import torch

from config import load_config
from src.data.bundle import load_bundle
from src.eval.evaluate import evaluate_caser
from src.models.caser import Caser, activation_getter
from src.train.common import (
    PAPER_REF,
    pick_device,
    print_comparison,
    resolve_category,
    run_config,
    save_run,
    set_seed,
)


def build_batches(split, L, batch_size, shuffle, device):
    """
    每行一个训练实例(与 GAMER 判别式基线一致):
    输入为 history 的最后 L 个物品(左侧补 0), 目标为该行 target。
    """
    indices = list(range(len(split)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        seq = torch.zeros(len(chunk), L, dtype=torch.long)
        targets = torch.zeros(len(chunk), dtype=torch.long)
        for i, j in enumerate(chunk):
            hist = split.histories[j][-L:]
            seq[i, L - len(hist) :] = torch.tensor(hist) + 1
            targets[i] = split.targets[j]
        yield seq.to(device), targets.to(device)


def main(cfg=None):
    cfg = cfg or load_config("caser")
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    train_cfg, eval_cfg = cfg["training"], cfg["evaluation"]
    for key in ("ac_conv", "ac_fc"):
        if model_cfg[key] not in activation_getter:
            raise ValueError(
                f"{key}={model_cfg[key]!r} 无效, "
                f"可选: {sorted(activation_getter)}"
            )
    category = resolve_category(data_cfg)
    run = run_config(cfg, category)

    set_seed(train_cfg["seed"])
    device = pick_device()
    bundle = load_bundle(data_cfg["root"], category)
    n_items = bundle.n_items
    model = Caser(
        n_items=n_items,
        L=model_cfg["L"],
        d=model_cfg["d"],
        nv=model_cfg["nv"],
        nh=model_cfg["nh"],
        drop=model_cfg["drop"],
        ac_conv=model_cfg["ac_conv"],
        ac_fc=model_cfg["ac_fc"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["l2"],
    )

    out_dir = train_cfg["out"] or os.path.join("runs", f"{category}_caser")
    os.makedirs(out_dir, exist_ok=True)

    best_ndcg = -1.0
    best_valid = None
    best_state = None
    bad_evals = 0
    t0 = time.time()

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss, n_batches = 0.0, 0
        for seq, targets in build_batches(
            bundle.train,
            model_cfg["L"],
            train_cfg["batch_size"],
            True,
            device,
        ):
            optimizer.zero_grad()
            loss = model.ce_loss(seq, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % train_cfg["eval_every"] == 0:
            valid_metrics = evaluate_caser(
                model,
                bundle.valid,
                device,
                k_list=tuple(eval_cfg["k_list"]),
                batch_size=eval_cfg["batch_size"],
                max_len=model_cfg["L"],
            )
            marker = ""
            if valid_metrics["NDCG@10"] > best_ndcg:
                best_ndcg = valid_metrics["NDCG@10"]
                best_valid = valid_metrics
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad_evals = 0
                marker = " *"
            else:
                bad_evals += 1
            print(
                f"epoch {epoch+1:>4} loss {total_loss/max(n_batches,1):.4f} "
                f"valid N@10 {valid_metrics['NDCG@10']:.4f} "
                f"R@10 {valid_metrics['Recall@10']:.4f} "
                f"({time.time()-t0:.0f}s){marker}",
                flush=True,
            )
            if bad_evals >= train_cfg["patience"]:
                print("early stop")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_caser(
        model,
        bundle.test,
        device,
        k_list=tuple(eval_cfg["k_list"]),
        batch_size=eval_cfg["batch_size"],
        max_len=model_cfg["L"],
    )
    torch.save(
        {
            "state_dict": best_state,
            "config": run,
            "n_items": n_items,
        },
        os.path.join(out_dir, "best.pt"),
    )
    print_comparison("Caser", category, test_metrics)
    save_run(
        out_dir,
        run,
        best_valid,
        test_metrics,
        PAPER_REF[(category, "Caser")],
    )


if __name__ == "__main__":
    main()
