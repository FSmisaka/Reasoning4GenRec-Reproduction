"""
SID 集切换(官方 <-> 自建), 用于 A/B 对比测试

启动: DATASET=games SID=official make games-sid-switch
      SID 可选 official(官方 RQ-VAE 产物) / own(本仓库 RQ-VAE 产物)

原理: SID 只通过 4 处文件进入下游(index.json / 3 个 CSV / info txt /
RL parquet), 代码与词表零耦合。两套完整快照:
- data/raw_official_backup/<Category>/  官方原件(rqvae_apply 首次备份)
- data/raw_own_backup/<Category>/        自建版本(本脚本自动维护)
切换时先把当前 in-place 文件存回其所属快照(保持最新), 再把目标
快照拷入 data/raw/Amazon/。

注意: 已训练的 TIGER / SIDReasoner checkpoint 与 SID 分配绑定,
切换后需按新 SID 重新训练再评估, 否则结果无效。
"""

import json
import os
import shutil

from config import load_config
from src.data.rqvae_embed import output_dir
from src.train.common import resolve_category

# in-place 路径 -> 快照内文件名(与 rqvae_apply 的备份命名一致)
FILES = {
    "index/{category}.index.json": "{category}.index.json",
    "train/{csv}": "train{suffix}.csv",
    "valid/{csv}": "valid{suffix}.csv",
    "test/{csv}": "test{suffix}.csv",
    "info/{txt}": "info{suffix}.txt",
    "rec_reasoning_verl/{category}/train.parquet": "verl_train.parquet",
    "rec_reasoning_verl/{category}/test.parquet": "verl_test.parquet",
}


def _paths(root, category, suffix):
    csv = f"{category}{suffix}.csv"
    txt = f"{category}{suffix}.txt"
    fmt = {"category": category, "csv": csv, "txt": txt, "suffix": suffix}
    in_place = {
        os.path.join(root, k.format(**fmt)): v.format(**fmt)
        for k, v in FILES.items()
    }
    return in_place


def _load_index(path):
    with open(path) as f:
        return json.load(f)


def detect_current(root, category, sets_dir):
    """比对 in-place index.json 与各快照, 判定当前生效的 SID 集。"""
    current = _load_index(os.path.join(root, "index", f"{category}.index.json"))
    for name in ("official", "own"):
        snapshot_index = os.path.join(
            sets_dir[name], category, f"{category}.index.json"
        )
        if os.path.exists(snapshot_index) and _load_index(snapshot_index) == current:
            return name
    own_src = os.path.join(output_dir(category), "index.own.json")
    if os.path.exists(own_src) and _load_index(own_src) == current:
        return "own"
    raise RuntimeError(
        "无法判定当前 SID 集: in-place index.json 与任何快照都不一致"
    )


def main(cfg=None):
    cfg = cfg or load_config("rqvae")
    data_cfg = cfg["data"]
    category = resolve_category(data_cfg)
    root = data_cfg["root"]
    suffix = data_cfg["suffix"]
    target = os.environ.get("SID")
    if target not in ("official", "own"):
        raise SystemExit("需要 SID=official 或 SID=own, 如: "
                         "DATASET=games SID=official make games-sid-switch")

    sets_dir = {
        "official": os.path.join("data", "raw_official_backup"),
        "own": os.path.join("data", "raw_own_backup"),
    }
    in_place = _paths(root, category, suffix)
    current = detect_current(root, category, sets_dir)
    print(f"[switch] 当前 SID 集: {current}, 目标: {target}")
    if current == target:
        print(f"[switch] 已是 {target}, 无需切换")
        return

    # 1) 把 in-place 存回当前集合的快照(首次 own 快照在此自动建立)
    for src, name in in_place.items():
        if not os.path.exists(src):
            continue
        dst = os.path.join(sets_dir[current], category, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    # 2) 目标快照 -> in-place
    missing = []
    for src, name in in_place.items():
        snapshot = os.path.join(sets_dir[target], category, name)
        if not os.path.exists(snapshot):
            missing.append(name)
            continue
        shutil.copy2(snapshot, src)
    if missing:
        raise SystemExit(f"[switch] {target} 快照缺少文件: {missing}")

    # 3) 验证
    from src.data.bundle import load_bundle

    bundle = load_bundle(root, category)
    assert detect_current(root, category, sets_dir) == target
    print(
        f"[switch] 已切换到 {target}: bundle n_items={bundle.n_items}, "
        f"train/valid/test={len(bundle.train)}/{len(bundle.valid)}"
        f"/{len(bundle.test)} 验证通过"
    )


if __name__ == "__main__":
    main()
