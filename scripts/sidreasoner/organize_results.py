"""一次性整理: 把旧版平铺在 results/sidreasoner/ 下的结果迁入两级目录。

按当前平铺文件的实际命名(含手动重命名)归类, 数据集从文件名推断:

  final_result_<C>.<label>.json               -> <D>/nonthink/<label'>.json
      label: stage1 / stage2 / stage3_step800 / ... (非思考模式评测)
  final_result_<C>[_own].json                 -> <D>/nonthink/legacy[_own].json
      (旧默认命名, 无模型信息)
  <C>_think_stage3_step100.json               -> <D>/think_steprule/step_100.json
      (思考模式贪心评测, label 形如 stageX_stepN)
  <C>_think_stage3_temp_step1900.json         -> <D>/think_steprule_temp/step_1900_summary.json
      (手动重命名过的温度采样 summary)
  <C>_think_<label>[_own]_seed<N>.json        -> <D>/think_steprule_temp/<label'>[_own]_seed<N>.json
  <C>_think_<label>[_own]_summary.json        -> <D>/think_steprule_temp/<label'>[_own]_summary.json
  <C>_think_<label>[_own].json                -> <D>/think_steprule/<label'>[_own].json
      (思考模式旧默认命名)
  <C>_le<N>_<mode>_<label>.json               -> <D>/le<N>/<mode>_<label'>.json
  <C>_<label>.json                            -> <D>/nonthink/<label'>.json
      (非思考旧默认命名)

label' 映射: stage3_step800 -> step_800; stage1/stage2 原样; 其余原样
(如 actor_merged 无法还原 step 数, 保留原名便于人工辨认)。

注意: 旧文件名不含训练变体信息, think 类默认归 think_steprule;
think_rolloutbeam / think_rolloutbeam_div 的结果出来后直接落新目录,
无需本脚本处理。同名冲突时跳过并警告(不覆盖)。

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
_CATS = "|".join(CAT2DIR)

CAT_RE = re.compile(r"^(" + _CATS + r")_(.+)$")
# final_result_<C>.<label>.json(手动重命名过, 点分隔, 含模型/step 信息)
FINAL_DOT_RE = re.compile(rf"^final_result_({_CATS})\.(?P<label>.+?)\.json$")
# final_result_<C>[_own].json(旧默认命名, 无模型信息)
FINAL_OLD_RE = re.compile(rf"^final_result_({_CATS})(?P<sfx>_own)?\.json$")
# <C>_le<N>_<mode>_<label>.json
LE_RE = re.compile(r"^le(?P<n>\d+)_(?P<mode>eval|think)_(?P<label>.+?)\.json$")
# think 产物的后缀
SEED_RE = re.compile(r"^(?P<base>.+?)(?P<own>_own)?_(?P<seed>seed\d+)$")
SUMMARY_RE = re.compile(r"^(?P<base>.+?)(?P<own>_own)?_summary$")
# 手动命名中的温度采样 summary: stage3_temp_step1900
TEMP_STEP_RE = re.compile(r"^stage\d+_temp_step(?P<n>\d+)$")
# 手动命名中的 step 标签: stage3_step800
STAGE_STEP_RE = re.compile(r"^stage\d+_step(?P<n>\d+)$")


def map_label(label: str) -> str:
    """把手动命名里的标签映射成新命名(stage3_step800 -> step_800)。"""
    if STAGE_STEP_RE.match(label):
        return f"step_{STAGE_STEP_RE.match(label).group('n')}"
    return label  # stage1/stage2/actor_merged/... 原样


def _strip_own(tail: str):
    """拆出末尾的 _own 后缀, 返回 (主体, own)。"""
    if tail.endswith("_own"):
        return tail[: -len("_own")], "_own"
    return tail, ""


def classify(fname):
    """返回 (dataset_dir, 相对新文件名) 或 None(无法识别)。"""
    m = FINAL_DOT_RE.match(fname)
    if m:
        label, own = _strip_own(m.group("label"))
        return CAT2DIR[m.group(1)], os.path.join("nonthink", f"{map_label(label)}{own}.json")

    m = FINAL_OLD_RE.match(fname)
    if m:
        return CAT2DIR[m.group(1)], os.path.join("nonthink", f"legacy{m.group('sfx') or ''}.json")

    m = CAT_RE.match(fname)
    if not m:
        return None
    cat, rest = m.group(1), m.group(2)
    ds = CAT2DIR[cat]

    if rest.startswith("think_"):
        tail = rest[len("think_"):].removesuffix(".json")

        t = SEED_RE.match(tail)  # 温度采样: 单 seed(_own 由正则捕获)
        if t:
            base = map_label(t.group("base")) + (t.group("own") or "")
            return ds, os.path.join("think_steprule_temp", f"{base}_{t.group('seed')}.json")

        t = SUMMARY_RE.match(tail)  # 温度采样: summary
        if t:
            base = map_label(t.group("base")) + (t.group("own") or "")
            return ds, os.path.join("think_steprule_temp", f"{base}_summary.json")

        t = TEMP_STEP_RE.match(tail)  # 手动命名的温度采样 summary
        if t:
            return ds, os.path.join("think_steprule_temp", f"step_{t.group('n')}_summary.json")

        # 贪心思考模式
        label, own = _strip_own(tail)
        return ds, os.path.join("think_steprule", f"{map_label(label)}{own}.json")

    m = LE_RE.match(rest)
    if m:
        label, lown = _strip_own(m.group("label"))
        return ds, os.path.join(
            f"le{m.group('n')}", f"{m.group('mode')}_{map_label(label)}{lown}.json"
        )

    # 非思考旧默认命名: <C>_<label>.json
    label, own = _strip_own(rest.removesuffix(".json"))
    return ds, os.path.join("nonthink", f"{map_label(label)}{own}.json")


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
