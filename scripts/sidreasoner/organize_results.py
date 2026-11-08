"""一次性整理: 把旧版平铺在 results/sidreasoner/ 下的结果迁入两级目录。

旧命名 -> 新位置(数据集从文件名前缀推断):
  final_result_<C><sfx>.json                -> <Dataset>/nonthink/legacy<sfx>.json
  <C>_<label>.json(非思考默认命名)           -> <Dataset>/nonthink/<label>.json
  <C>_think_<label><sfx>.json               -> <Dataset>/think_steprule/<label><sfx>.json
  <C>_think_<label><sfx>_seed<N>.json       -> <Dataset>/think_steprule_temp/<label><sfx>_seed<N>.json
  <C>_think_<label><sfx>_summary.json       -> <Dataset>/think_steprule_temp/<label><sfx>_summary.json
  <C>_le<N>_<mode>_<label>.json             -> <Dataset>/le<N>/<mode>_<label>.json

注意: 旧文件名不含训练变体信息, think 类默认归 think_steprule;
若确知某文件来自 rollout_beam / rolloutbeam_div 模型, 请搬完后手动挪到
对应 think_rolloutbeam[_div] 目录。同名冲突时跳过并警告(不覆盖)。

用法: [DRY_RUN=1] .venv/bin/python scripts/sidreasoner/organize_results.py
"""

import os
import re
import shutil
import sys

sys.path.insert(0, ".")

from src.eval.result_routing import RESULT_ROOT  # noqa: E402

# category 前缀 -> Dataset 目录
CAT2DIR = {"Video_Games": "Games", "Office_Products": "Office",
           "Industrial_and_Scientific": "Industrial"}
CAT_RE = re.compile(
    r"^(" + "|".join(CAT2DIR) + r")_(.*)$"
)
THINK_RE = re.compile(r"^(?P<label>.+?)(?P<own>_own)?(?P<tail>_seed\d+|_summary)?\.json$")
LE_RE = re.compile(r"^le(?P<n>\d+)_(?P<mode>eval|think)_(?P<label>.+?)\.json$")


def classify(fname):
    """返回 (dataset_dir, 相对新文件名) 或 None(无法识别)。"""
    # final_result_<C><sfx>.json(category 在中间)
    m = re.match(r"^final_result_(" + "|".join(CAT2DIR) + r")(?P<sfx>_own)?\.json$", fname)
    if m:
        return CAT2DIR[m.group(1)], os.path.join("nonthink", f"legacy{m.group('sfx') or ''}.json")

    m = CAT_RE.match(fname)
    if not m:
        return None
    cat, rest = m.group(1), m.group(2)
    ds = CAT2DIR[cat]

    if rest.startswith("think_"):
        t = THINK_RE.match(rest[len("think_"):])
        if t:
            label, own, tail = t.group("label"), t.group("own") or "", t.group("tail") or ""
            sub = "think_steprule_temp" if tail else "think_steprule"
            return ds, os.path.join(sub, f"{label}{own}{tail}.json")
        return None

    m = LE_RE.match(rest)
    if m:
        return ds, os.path.join(
            f"le{m.group('n')}", f"{m.group('mode')}_{m.group('label')}.json"
        )

    # 非思考默认命名: <C>_<label>.json
    return ds, os.path.join("nonthink", f"{rest}")


def main():
    dry = os.environ.get("DRY_RUN", "0") == "1"
    root = RESULT_ROOT
    if not os.path.isdir(root):
        print(f"目录不存在: {root}")
        return
    moved = skipped = 0
    for fname in sorted(os.listdir(root)):
        src = os.path.join(root, fname)
        if not os.path.isfile(src) or not fname.endswith(".json"):
            continue
        hit = classify(fname)
        if hit is None:
            print(f"[skip] 无法识别: {fname}")
            skipped += 1
            continue
        ds, rel = hit
        dst = os.path.join(root, ds, rel)
        if os.path.exists(dst):
            print(f"[skip] 目标已存在(不覆盖): {fname} -> {ds}/{rel}")
            skipped += 1
            continue
        print(f"[{'dry' if dry else 'move'}] {fname} -> {ds}/{rel}")
        if not dry:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved += 1
    print(f"完成: 移动 {moved} 个, 跳过 {skipped} 个"
          + ("(DRY_RUN, 未实际移动)" if dry else ""))


if __name__ == "__main__":
    main()
