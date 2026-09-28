import os
import random
import time

import torch
from torch.nn import functional as F

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


def build_user_index(split):
    users = sorted(set(split.users))
    return {u: i + 1 for i, u in enumerate(users)}


def build_seen_items(split, user_index):
    seen = {}
    for j in range(len(split)):
        u = user_index[split.users[j]]
        items = set(x + 1 for x in split.histories[j])
        items.add(split.targets[j] + 1)
        seen.setdefault(u, set()).update(items)
    return seen


def build_instances(split, user_index, L, T):
    """
    对齐原始实现 Interactions.to_sequence / _sliding_window:
    对每个用户的完整序列 (history + target, 物品 id +1, 0 为 padding)
    滑动长度 L+T 的窗口, 取前 L 个为输入、后 T 个为目标;
    序列不足一个窗口时在左侧补 0 得到单个窗口。
    """
    window = L + T
    instances = []
    for j in range(len(split)):
        u = user_index.get(split.users[j], 0)
        full = [x + 1 for x in split.histories[j]] + [split.targets[j] + 1]
        if len(full) < window:
            windows = [[0] * (window - len(full)) + full]
        else:
            windows = [
                full[i : i + window]
                for i in range(len(full) - window + 1)
            ]
        for w in windows:
            instances.append((u, w[:L], w[L:]))
    return instances


def sample_negatives(exclude, shape, rounds):
    n = exclude.size(1)
    neg = torch.randint(1, n, shape, device=exclude.device)
    for _ in range(rounds):
        bad = exclude.gather(1, neg)
        if not bad.any():
            break
        neg = torch.where(
            bad,
            torch.randint(1, n, neg.shape, device=exclude.device),
            neg,
        )
    return neg


def build_batches(
    instances,
    seen,
    n_items,
    batch_size,
    neg_samples,
    neg_rounds,
    shuffle,
    device,
):
    indices = list(range(len(instances)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        B = len(chunk)
        L = len(instances[chunk[0]][1])
        T = len(instances[chunk[0]][2])
        users = torch.zeros(B, 1, dtype=torch.long)
        seq = torch.zeros(B, L, dtype=torch.long)
        targets = torch.zeros(B, T, dtype=torch.long)
        exclude = torch.zeros(B, n_items + 1, dtype=torch.bool)
        for i, idx in enumerate(chunk):
            u, s, t = instances[idx]
            users[i, 0] = u
            seq[i] = torch.tensor(s)
            targets[i] = torch.tensor(t)
            exclude[i, sorted(seen.get(u, ()))] = True
        exclude = exclude.to(device)
        neg = sample_negatives(exclude, (B, neg_samples), neg_rounds)
        yield (
            users.to(device),
            seq.to(device),
            targets.to(device),
            neg,
        )


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
    user_index = build_user_index(bundle.train)
    num_users = len(user_index) + 1
    seen = build_seen_items(bundle.train, user_index)
    model = Caser(
        num_users=num_users,
        num_items=n_items + 1,
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
    instances = build_instances(
        bundle.train, user_index, model_cfg["L"], model_cfg["T"]
    )
    print(f"total training instances: {len(instances)}")

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
        for users, seq, targets, neg in build_batches(
            instances,
            seen,
            n_items,
            train_cfg["batch_size"],
            train_cfg["neg_samples"],
            train_cfg["neg_resample_rounds"],
            True,
            device,
        ):
            items_to_predict = torch.cat((targets, neg), 1)
            items_prediction = model(seq, users, items_to_predict)
            (
                targets_prediction,
                negatives_prediction,
            ) = torch.split(
                items_prediction,
                [targets.size(1), neg.size(1)],
                dim=1,
            )
            optimizer.zero_grad()
            # -mean(log(sigmoid(pos))) - mean(log(1 - sigmoid(neg)))
            # 的数值稳定等价形式 (softplus)
            loss = F.softplus(targets_prediction).mean() + F.softplus(
                -negatives_prediction
            ).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % train_cfg["eval_every"] == 0:
            valid_metrics = evaluate_caser(
                model,
                bundle.valid,
                user_index,
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
        user_index,
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
            "num_users": num_users,
        },
        os.path.join(out_dir, "best.pt"),
    )
    print_comparison("Caser", category, test_metrics)
    save_run(
        out_dir,
        run,
        best_valid,
        test_metrics,
        PAPER_REF.get((category, "Caser")),
    )


if __name__ == "__main__":
    main()
