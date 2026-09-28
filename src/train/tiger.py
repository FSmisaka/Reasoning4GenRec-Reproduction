import os
import random
import time

import torch

from config import load_config
from src.data.bundle import load_bundle
from src.eval.evaluate import evaluate_tiger
from src.models.tiger import (
    build_tiger,
    encode_history,
    encode_target,
)
from src.train.common import (
    PAPER_REF,
    pick_device,
    print_comparison,
    resolve_category,
    run_config,
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


def main(cfg=None):
    cfg = cfg or load_config("tiger")
    data_cfg, model_cfg = cfg["data"], cfg["model"]
    train_cfg, eval_cfg = cfg["training"], cfg["evaluation"]
    gen_cfg = cfg["generation"]
    if train_cfg["optimizer"] not in train_cfg["optimizer_choices"]:
        raise ValueError(
            f"optimizer={train_cfg['optimizer']!r} 无效, "
            f"可选: {train_cfg['optimizer_choices']}"
        )
    if train_cfg["scheduler"] not in train_cfg["scheduler_choices"]:
        raise ValueError(
            f"scheduler={train_cfg['scheduler']!r} 无效, "
            f"可选: {train_cfg['scheduler_choices']}"
        )
    category = resolve_category(data_cfg)
    run = run_config(cfg, category)

    set_seed(train_cfg["seed"])
    device = pick_device()
    bundle = load_bundle(data_cfg["root"], category)
    sid_table = bundle.sid_table
    model = build_tiger(
        sid_table,
        d_model=model_cfg["d_model"],
        d_ff=model_cfg["d_ff"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        feed_forward_proj=model_cfg["feed_forward_proj"],
    ).to(device)
    if train_cfg["optimizer"] == "adafactor":
        from transformers.optimization import Adafactor

        optimizer = Adafactor(
            model.parameters(),
            lr=train_cfg["lr"],
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg["lr"]
        )
    if train_cfg["scheduler"] == "invsq":
        from transformers.optimization import get_inverse_sqrt_schedule

        scheduler = get_inverse_sqrt_schedule(
            optimizer, num_warmup_steps=train_cfg["warmup_steps"]
        )
    else:
        scheduler = None

    out_dir = train_cfg["out"] or os.path.join("runs", f"{category}_tiger")
    os.makedirs(out_dir, exist_ok=True)

    valid_split = bundle.valid
    if (
        train_cfg["eval_subset"]
        and train_cfg["eval_subset"] < len(valid_split)
    ):

        class Subset:
            pass

        sub = Subset()
        sub.hist_sids = valid_split.hist_sids[: train_cfg["eval_subset"]]
        sub.targets = valid_split.targets[: train_cfg["eval_subset"]]
        valid_eval = sub
    else:
        valid_eval = valid_split

    best_ndcg = -1.0
    best_valid = None
    best_state = None
    bad_evals = 0
    t0 = time.time()

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss, n_batches = 0.0, 0
        for input_ids, attention_mask, labels in build_batches(
            bundle.train,
            sid_table,
            train_cfg["batch_size"],
            train_cfg["max_items"],
            True,
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
        if (epoch + 1) % train_cfg["eval_every"] == 0:
            valid_metrics = evaluate_tiger(
                model,
                valid_eval,
                sid_table,
                device,
                k_list=tuple(eval_cfg["k_list"]),
                num_beams=gen_cfg["num_beams"],
                batch_size=gen_cfg["batch_size"],
                max_len=train_cfg["max_items"],
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
            if bad_evals >= train_cfg["patience"]:
                print("early stop")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_tiger(
        model,
        bundle.test,
        sid_table,
        device,
        k_list=tuple(eval_cfg["k_list"]),
        num_beams=gen_cfg["num_beams"],
        batch_size=gen_cfg["batch_size"],
        max_len=train_cfg["max_items"],
    )
    torch.save(
        {"state_dict": best_state, "config": run},
        os.path.join(out_dir, "best.pt"),
    )
    print_comparison("TIGER", category, test_metrics)
    save_run(
        out_dir,
        run,
        best_valid,
        test_metrics,
        PAPER_REF[(category, "TIGER")],
    )


if __name__ == "__main__":
    main()
