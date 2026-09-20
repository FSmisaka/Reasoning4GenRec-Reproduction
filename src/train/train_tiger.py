import argparse
import os
import random
import time

import torch

from src.data.bundle import load_bundle
from src.eval.evaluate import evaluate_tiger
from src.models.tiger.model import (
    build_tiger,
    encode_history,
    encode_target,
)
from src.train.common import (
    PAPER_REF,
    pick_device,
    print_comparison,
    save_run,
    set_seed,
)


def build_batches(split, sid_table, batch_size, max_items, shuffle):
    indices = list(range(len(split)))
    if shuffle:
        random.shuffle(indices)
    max_tokens = max_items * sid_table.num_levels
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        enc, dec = [], []
        for j in chunk:
            enc.append(
                encode_history(split.hist_sids[j], sid_table, max_tokens)
            )
            dec.append(encode_target(split.tgt_sids[j], sid_table))
        L = max(len(t) for t in enc)
        input_ids = torch.zeros(len(chunk), L, dtype=torch.long)
        attention_mask = torch.zeros(len(chunk), L, dtype=torch.long)
        for i, toks in enumerate(enc):
            input_ids[i, : len(toks)] = torch.tensor(toks)
            attention_mask[i, : len(toks)] = 1
        labels = torch.tensor(dec, dtype=torch.long)
        yield input_ids, attention_mask, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--data-root", default="data/raw/Amazon")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--num-beams", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-subset", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--scheduler", default="constant", choices=["constant", "invsq"]
    )
    parser.add_argument(
        "--optimizer", default="adam", choices=["adam", "adafactor"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    bundle = load_bundle(args.data_root, args.category)
    sid_table = bundle.sid_table
    model = build_tiger(
        sid_table,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor

        optimizer = Adafactor(
            model.parameters(),
            lr=args.lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == "invsq":
        from transformers.optimization import get_inverse_sqrt_schedule

        scheduler = get_inverse_sqrt_schedule(
            optimizer, num_warmup_steps=args.warmup_steps
        )
    else:
        scheduler = None

    out_dir = args.out or os.path.join("runs", f"{args.category}_tiger")
    os.makedirs(out_dir, exist_ok=True)

    valid_split = bundle.valid
    if args.eval_subset and args.eval_subset < len(valid_split):

        class Subset:
            pass

        sub = Subset()
        sub.hist_sids = valid_split.hist_sids[: args.eval_subset]
        sub.targets = valid_split.targets[: args.eval_subset]
        valid_eval = sub
    else:
        valid_eval = valid_split

    best_ndcg = -1.0
    best_valid = None
    best_state = None
    bad_evals = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for input_ids, attention_mask, labels in build_batches(
            bundle.train, sid_table, args.batch_size, args.max_items, True
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            out.loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += out.loss.item()
            n_batches += 1
        if (epoch + 1) % args.eval_every == 0:
            valid_metrics = evaluate_tiger(
                model,
                valid_eval,
                sid_table,
                device,
                num_beams=args.num_beams,
                max_len=args.max_items,
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
                f"valid(sub) N@10 {valid_metrics['NDCG@10']:.4f} "
                f"R@10 {valid_metrics['Recall@10']:.4f} "
                f"({time.time()-t0:.0f}s){marker}",
                flush=True,
            )
            if bad_evals >= args.patience:
                print("early stop")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_tiger(
        model,
        bundle.test,
        sid_table,
        device,
        num_beams=args.num_beams,
        max_len=args.max_items,
    )
    torch.save(
        {"state_dict": best_state, "config": vars(args)},
        os.path.join(out_dir, "best.pt"),
    )
    print_comparison("TIGER", args.category, test_metrics)
    save_run(
        out_dir,
        vars(args),
        best_valid,
        test_metrics,
        PAPER_REF[(args.category, "TIGER")],
    )


if __name__ == "__main__":
    main()
