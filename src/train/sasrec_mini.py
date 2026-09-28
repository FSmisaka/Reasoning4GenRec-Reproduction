import ast
import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import load_config
from src.eval.metrics import rank_metrics, rank_of
from src.models.sasrec_mini import SASRecMiniOneRec
from src.train.common import (
    PAPER_REF,
    pick_device,
    print_comparison,
    resolve_category,
    run_config,
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
def evaluate_model(
    model, split_df, item_num, seq_size, device, k_list, batch_size
):
    model.eval()
    seqs, lens, targets = prepare_frame(split_df, item_num, seq_size)
    ranks = []
    for start in range(0, len(targets), batch_size):
        logits = model(
            seqs[start : start + batch_size].to(device),
            lens[start : start + batch_size].to(device),
        )
        _, topk = torch.topk(logits, max(k_list), dim=1)
        topk = topk.cpu().tolist()
        for i, tgt in enumerate(targets[start : start + batch_size].tolist()):
            ranks.append(rank_of(tgt, topk[i]))
    model.train()
    return rank_metrics(ranks, k_list)


def main(cfg=None):
    cfg = cfg or load_config("sasrec_mini")
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    train_cfg, eval_cfg = cfg["training"], cfg["evaluation"]
    category = resolve_category(data_cfg)
    run = run_config(cfg, category)

    set_seed(train_cfg["seed"])
    device = pick_device()

    data_root = data_cfg["root"]
    fname = f"{category}_5_2016-10-2018-11.csv"
    train_df = pd.read_csv(os.path.join(data_root, "train", fname))
    valid_df = pd.read_csv(os.path.join(data_root, "valid", fname))
    test_df = pd.read_csv(os.path.join(data_root, "test", fname))
    info_path = os.path.join(
        data_root, "info", f"{category}_5_2016-10-2018-11.txt"
    )
    item_num = len(open(info_path).readlines())
    seq_size = model_cfg["seq_size"]

    train_seqs, train_lens, train_targets = prepare_frame(
        train_df, item_num, seq_size
    )
    loader = DataLoader(
        RecDataset(train_seqs, train_lens, train_targets),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )

    model = SASRecMiniOneRec(
        model_cfg["hidden_factor"],
        item_num,
        seq_size,
        model_cfg["dropout"],
        model_cfg["num_heads"],
        model_cfg["init_std"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        eps=train_cfg["adam_eps"],
        weight_decay=train_cfg["l2_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss()

    out_dir = train_cfg["out"] or os.path.join(
        "runs", f"{category}_sasrec_mini"
    )
    os.makedirs(out_dir, exist_ok=True)

    k_list = tuple(eval_cfg["k_list"])
    best_key = f"NDCG@{max(k_list)}"

    best_ndcg = -1.0
    best_valid = None
    best_state = None
    best_epoch = 0
    early_stop = 0
    t0 = time.time()

    for epoch in range(train_cfg["epochs"]):
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
            model,
            valid_df,
            item_num,
            seq_size,
            device,
            k_list,
            eval_cfg["batch_size"],
        )
        ndcg20 = valid_metrics[best_key]
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
        if early_stop > train_cfg["patience"]:
            print(f"early stop at epoch {epoch}, best epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(
        model, test_df, item_num, seq_size, device, k_list,
        eval_cfg["batch_size"],
    )
    torch.save(
        {"state_dict": best_state, "config": run, "item_num": item_num},
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
