import argparse
import os
import time

import torch

from src.data.bundle import load_bundle
from src.eval.evaluate import evaluate_sasrec
from src.models.sasrec.model import SASRec
from src.train.common import (
    PAPER_REF,
    pick_device,
    print_comparison,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument(
        "--data-root", default="data/raw/Amazon"
    )
    parser.add_argument("--loss", default="bce", choices=["bce", "ce"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    bundle = load_bundle(args.data_root, args.category)
    n_items = bundle.n_items
    model = SASRec(
        n_items=n_items,
        max_seq_len=args.max_len,
        hidden=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = args.out or os.path.join(
        "runs", f"{args.category}_sasrec_{args.loss}"
    )
    os.makedirs(out_dir, exist_ok=True)

    best_ndcg = -1.0
    best_valid = None
    best_state = None
    bad_evals = 0
    step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for seq, labels, exclude in build_batches(
            bundle.train, n_items, args.max_len, args.batch_size, True, device
        ):
            optimizer.zero_grad()
            if args.loss == "bce":
                loss = model.bce_loss(seq, labels, exclude)
            else:
                loss = model.ce_loss(seq, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            step += 1
        if (epoch + 1) % args.eval_every == 0:
            valid_metrics = evaluate_sasrec(
                model, bundle.valid, device, batch_size=512, max_len=args.max_len
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
            if bad_evals >= args.patience:
                print("early stop")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_sasrec(
        model, bundle.test, device, batch_size=512, max_len=args.max_len
    )
    torch.save(
        {
            "state_dict": best_state,
            "config": vars(args),
            "n_items": n_items,
        },
        os.path.join(out_dir, "best.pt"),
    )
    print_comparison("SASRec", args.category, test_metrics)
    save_run(
        out_dir,
        vars(args),
        best_valid,
        test_metrics,
        PAPER_REF[(args.category, "SASRec")],
    )


if __name__ == "__main__":
    main()
