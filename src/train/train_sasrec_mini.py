import argparse
import ast
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.bundle import load_bundle
from src.eval.metrics import rank_metrics, rank_of
from src.models.sasrec.minionerec import SASRecMiniOneRec
from src.train.common import (
    PAPER_REF,
    pick_device,
    print_comparison,
    save_run,
    set_seed,
)


class RecDataset(Dataset):
    def __init__(self, seqs, lens, targets):
        self.seqs = seqs
        self.lens = lens
        self.targets = targets

    def __getitem__(self, i):
        return self.seqs[i], self.lens[i], self.targets[i]

    def __len__(self):
        return len(self.targets)


def prepare_frame(df, item_num, seq_size):
    seqs = [ast.literal_eval(s) for s in df["history_item_id"]]
    lens = [len(s) for s in seqs]
    seqs = [s + [item_num] * (seq_size - len(s)) for s in seqs]
    targets = [int(t) for t in df["item_id"]]
    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(lens, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


@torch.no_grad()
def evaluate_model(model, split_df, item_num, seq_size, device, k_list=(5, 10, 20)):
    model.eval()
    seqs, lens, targets = prepare_frame(split_df, item_num, seq_size)
    ranks = []
    bs = 1024
    for start in range(0, len(targets), bs):
        logits = model(
            seqs[start : start + bs].to(device),
            lens[start : start + bs].to(device),
        )
        _, topk = torch.topk(logits, max(k_list), dim=1)
        topk = topk.cpu().tolist()
        for i, tgt in enumerate(targets[start : start + bs].tolist()):
            ranks.append(rank_of(tgt, topk[i]))
    model.train()
    return rank_metrics(ranks, k_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="Video_Games")
    parser.add_argument("--data-root", default="data/raw/Amazon")
    parser.add_argument("--hidden-factor", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seq-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()

    data_root = args.data_root
    fname = f"{args.category}_5_2016-10-2018-11.csv"
    train_df = pd.read_csv(os.path.join(data_root, "train", fname))
    valid_df = pd.read_csv(os.path.join(data_root, "valid", fname))
    test_df = pd.read_csv(os.path.join(data_root, "test", fname))
    info_path = os.path.join(
        data_root, "info", f"{args.category}_5_2016-10-2018-11.txt"
    )
    item_num = len(open(info_path).readlines())
    seq_size = args.seq_size

    train_seqs, train_lens, train_targets = prepare_frame(
        train_df, item_num, seq_size
    )
    loader = DataLoader(
        RecDataset(train_seqs, train_lens, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = SASRecMiniOneRec(
        args.hidden_factor, item_num, seq_size, args.dropout, args.num_heads
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    out_dir = args.out or os.path.join(
        "runs", f"{args.category}_sasrec_minionerec"
    )
    os.makedirs(out_dir, exist_ok=True)

    best_ndcg = -1.0
    best_valid = None
    best_state = None
    best_epoch = 0
    early_stop = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        total_loss, n_batches = 0.0, 0
        for seq, len_seq, target in loader:
            target_neg = []
            for t in target.tolist():
                neg = np.random.randint(item_num)
                while neg == t:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)
            seq = seq.to(device)
            len_seq = len_seq.to(device)
            target = target.to(device)
            target_neg = torch.tensor(target_neg, device=device)
            optimizer.zero_grad()
            logits = model(seq, len_seq)
            pos_scores = torch.gather(logits, 1, target.view(-1, 1))
            neg_scores = torch.gather(logits, 1, target_neg.view(-1, 1))
            scores = torch.cat((pos_scores, neg_scores), 0).squeeze(-1)
            labels = torch.cat(
                (torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), 0
            ).squeeze(-1)
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        valid_metrics = evaluate_model(
            model, valid_df, item_num, seq_size, device
        )
        ndcg20 = valid_metrics["NDCG@20"]
        marker = ""
        if ndcg20 > best_ndcg:
            best_ndcg = ndcg20
            best_valid = valid_metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            early_stop = 0
            marker = " *"
        else:
            early_stop += 1
        if (epoch + 1) % 5 == 0 or marker:
            print(
                f"epoch {epoch+1:>4} loss {total_loss/max(n_batches,1):.4f} "
                f"valid N@5 {valid_metrics['NDCG@5']:.4f} "
                f"N@10 {valid_metrics['NDCG@10']:.4f} "
                f"({time.time()-t0:.0f}s){marker}",
                flush=True,
            )
        if early_stop > args.patience:
            print(f"early stop at epoch {epoch}, best epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_df, item_num, seq_size, device)
    torch.save(
        {"state_dict": best_state, "config": vars(args), "item_num": item_num},
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
