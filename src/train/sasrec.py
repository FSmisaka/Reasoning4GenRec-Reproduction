import os
import time

import torch

from config import load_config
from src.data.bundle import load_bundle
from src.eval.evaluate import evaluate_sasrec
from src.models.sasrec import SASRec
from src.train.common import (
    PAPER_REF,
    pick_device,
    print_comparison,
    resolve_category,
    run_config,
    save_run,
    set_seed,
)


def build_batches(split, n_items, max_len, batch_size, shuffle, device):
    indices = list(range(len(split)))
    if shuffle:
        import random

        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        seq = torch.zeros(len(chunk), max_len, dtype=torch.long)
        labels = torch.zeros(len(chunk), max_len, dtype=torch.long)
        exclude = torch.zeros(len(chunk), n_items + 1, dtype=torch.bool)
        for i, j in enumerate(chunk):
            hist = split.histories[j]
            tgt = split.targets[j]
            full = [x + 1 for x in hist] + [tgt + 1]
            inp = full[:-1]
            lab = full[1:]
            seq[i, max_len - len(inp) :] = torch.tensor(inp)
            labels[i, max_len - len(lab) :] = torch.tensor(lab)
            banned = set(full) | {0}
            exclude[i, list(banned)] = True
        yield seq.to(device), labels.to(device), exclude.to(device)


def main(cfg=None):
    cfg = cfg or load_config("sasrec")
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    train_cfg, eval_cfg = cfg["training"], cfg["evaluation"]
    if train_cfg["loss"] not in train_cfg["loss_choices"]:
        raise ValueError(
            f"loss={train_cfg['loss']!r} 无效,可选: {train_cfg['loss_choices']}"
        )
    if train_cfg["loss"] not in train_cfg["loss_choices"]:
        raise ValueError(
            f"loss={train_cfg['loss']!r} 无效,可选: {train_cfg['loss_choices']}"
        )
    category = resolve_category(data_cfg)
    run = run_config(cfg, category)

    set_seed(train_cfg["seed"])
    device = pick_device()
    bundle = load_bundle(data_cfg["root"], category)
    n_items = bundle.n_items
    model = SASRec(
        n_items=n_items,
        max_seq_len=model_cfg["max_seq_len"],
        hidden=model_cfg["hidden"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_inner=model_cfg["d_inner"],
        dropout=model_cfg["dropout"],
        init_std=model_cfg["init_std"],
        layer_norm_eps=model_cfg["layer_norm_eps"],
        neg_resample_rounds=model_cfg["neg_resample_rounds"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["l2"],
    )

    out_dir = train_cfg["out"] or os.path.join(
        "runs", f"{category}_sasrec_{train_cfg['loss']}"
    )
    os.makedirs(out_dir, exist_ok=True)

    best_ndcg = -1.0
    best_valid = None
    best_state = None
    bad_evals = 0
    step = 0
    t0 = time.time()

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss, n_batches = 0.0, 0
        for seq, labels, exclude in build_batches(
            bundle.train,
            n_items,
            model_cfg["max_seq_len"],
            train_cfg["batch_size"],
            True,
            device,
        ):
            optimizer.zero_grad()
            if train_cfg["loss"] == "bce":
                loss = model.bce_loss(seq, labels, exclude)
            elif train_cfg["loss"] == "bce1":
                loss = model.bce_last_loss(
                    seq, labels, n_neg=train_cfg["neg_samples"]
                )
            else:
                loss = model.ce_loss(seq, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            step += 1
        if (epoch + 1) % train_cfg["eval_every"] == 0:
            valid_metrics = evaluate_sasrec(
                model,
                bundle.valid,
                device,
                k_list=tuple(eval_cfg["k_list"]),
                batch_size=eval_cfg["batch_size"],
                max_len=model_cfg["max_seq_len"],
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
    test_metrics = evaluate_sasrec(
        model,
        bundle.test,
        device,
        k_list=tuple(eval_cfg["k_list"]),
        batch_size=eval_cfg["batch_size"],
        max_len=model_cfg["max_seq_len"],
    )
    torch.save(
        {
            "state_dict": best_state,
            "config": run,
            "n_items": n_items,
        },
        os.path.join(out_dir, "best.pt"),
    )
    print_comparison("SASRec", category, test_metrics)
    save_run(
        out_dir,
        run,
        best_valid,
        test_metrics,
        PAPER_REF[(category, "SASRec")],
    )


if __name__ == "__main__":
    main()
