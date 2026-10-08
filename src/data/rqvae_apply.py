"""
RQ-VAE 第 3 步: 级联替换官方 SID 文件(带一次性备份)

启动: DATASET=games make games-rqvae-apply
      (等价: DATASET=games .venv/bin/python -m src.data.rqvae_apply)
前置: 先跑 src.train.rqvae 生成 runs/rqvae/<Category>/index.own.json

动作:
1. 把将被修改的官方文件备份到 data/raw/Amazon_official_backup/<Category>/
   (仅首次, 备份永远保留最初的官方版本, 幂等);
2. 覆盖 index/<Category>.index.json;
3. 重写 {train,valid,test}/<Category>_5_*.csv 的 history_item_sid /
   item_sid 两列(按 item_id 查新 SID, 格式与官方一致);
4. 重写 info/<Category>_5_*.txt(SID\ttitle\titem_id);
5. 重建 rec_reasoning_verl/<Category>/{train,test}.parquet
   (复用 src.data.sidreasoner_rl_data)。

注意: 重建 parquet 基于截断为 10 条历史的 CSV, 比官方 parquet 的
历史(均值 ~15.6)短, 这是上游已知差异, 但 SID 保证了与三阶段训练
的一致性。item.json / item_enhanced_v2.json / integrated_narrative.csv
不含 SID 字符串, 无需改动。
"""

import ast
import json
import os
import shutil

import pandas as pd

from config import load_config
from src.data.sidreasoner_rl_data import main as rebuild_rl_data
from src.data.rqvae_embed import output_dir
from src.train.common import resolve_category


def _load_own_index(out_dir):
    path = os.path.join(out_dir, "index.own.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 {path}, 先运行: "
            "DATASET=<dataset> make <dataset>-rqvae-train"
        )
    raw = json.load(open(path))
    tokens = {int(k): list(v) for k, v in raw.items()}
    joined = {i: "".join(v) for i, v in tokens.items()}
    return tokens, joined


def _backup_once(src, dst):
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _write_index_json(item2tokens, path):
    entries = sorted(item2tokens.items())
    with open(path, "w") as f:
        f.write("{\n")
        for i, (k, v) in enumerate(entries):
            comma = "," if i < len(entries) - 1 else ""
            f.write(f"{json.dumps(str(k))}: {json.dumps(v)}{comma}\n")
        f.write("}\n")


def _rewrite_csv(csv_path, item2sid):
    df = pd.read_csv(csv_path)
    missing = {int(t) for t in df["item_id"]} - set(item2sid)
    if missing:
        raise KeyError(
            f"{csv_path} 中的 item_id 缺少新 SID: {sorted(missing)[:5]}"
        )

    df["item_sid"] = [item2sid[int(t)] for t in df["item_id"]]
    df["history_item_sid"] = [
        repr([item2sid[int(i)] for i in ast.literal_eval(ids)])
        if isinstance(ids, str)
        else repr([])
        for ids in df["history_item_id"]
    ]
    df.to_csv(csv_path, index=False)


def main(cfg=None):
    cfg = cfg or load_config("rqvae")
    data_cfg = cfg["data"]
    category = resolve_category(data_cfg)
    root = data_cfg["root"]
    suffix = data_cfg["suffix"]
    csv_name = f"{category}{suffix}.csv"
    out_dir = output_dir(category)

    item2tokens, item2sid = _load_own_index(out_dir)
    index_path = os.path.join(root, "index", f"{category}.index.json")
    old_index = {int(k): v for k, v in json.load(open(index_path)).items()}
    if set(old_index) != set(item2tokens):
        raise RuntimeError(
            "新 SID 表与官方 index.json 的 item 集合不一致, "
            f"{len(set(old_index) ^ set(item2tokens))} 个差异 item"
        )
    overlap = sum(
        "".join(old_index[i]) == item2sid[i] for i in item2sid
    )

    backup_dir = os.path.join(
        os.path.dirname(root.rstrip("/")) + "_official_backup", category
    )
    plan = {
        index_path: f"{category}.index.json",
        os.path.join(root, "train", csv_name): f"train{suffix}.csv",
        os.path.join(root, "valid", csv_name): f"valid{suffix}.csv",
        os.path.join(root, "test", csv_name): f"test{suffix}.csv",
        os.path.join(
            root, "info", f"{category}{suffix}.txt"
        ): f"info{suffix}.txt",
        os.path.join(
            root, "rec_reasoning_verl", category, "train.parquet"
        ): "verl_train.parquet",
        os.path.join(
            root, "rec_reasoning_verl", category, "test.parquet"
        ): "verl_test.parquet",
    }
    for src, name in plan.items():
        _backup_once(src, os.path.join(backup_dir, name))
    print(f"[apply] 官方原件备份目录: {backup_dir}")

    _write_index_json(item2tokens, index_path)
    print(f"[apply] 已覆盖 index/{category}.index.json")

    for split in ("train", "valid", "test"):
        _rewrite_csv(os.path.join(root, split, csv_name), item2sid)
        print(f"[apply] 已重写 {split}/{csv_name}")

    items = json.load(
        open(os.path.join(root, "index", f"{category}.item.json"))
    )
    info_path = os.path.join(root, "info", f"{category}{suffix}.txt")
    with open(info_path, "w") as f:
        for k in sorted(items, key=int):
            title = (items[k].get("title") or "").replace("\t", " ")
            f.write(f"{item2sid[int(k)]}\t{title}\t{k}\n")
    print(f"[apply] 已重写 info/{category}{suffix}.txt")

    rebuild_rl_data()
    print(
        "[apply] 已重建 rec_reasoning_verl/ (基于 CSV 的历史截断为 10, "
        "比官方 parquet 短, 属上游已知差异)"
    )

    from src.data.bundle import load_bundle

    bundle = load_bundle(root, category)
    assert bundle.n_items == len(item2sid)
    print(
        f"[apply] 验证通过: bundle n_items={bundle.n_items}, "
        f"train/valid/test={len(bundle.train)}/{len(bundle.valid)}"
        f"/{len(bundle.test)}, 与官方 SID 完全一致的比例 "
        f"{overlap / len(item2sid):.4f} (独立训练, 接近随机属正常)"
    )


if __name__ == "__main__":
    main()
