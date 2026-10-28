"""
SIDReasoner 短历史子集评估(思考 vs 非思考对照)

动机: ReaRec 指出"思考"对交互历史较短的用户更有帮助。本脚本把
测试集筛出 history_item_sid 长度 <= N(默认 3)的子集, 然后原样
调用现有两条评测管线(不修改它们), 分别得到两种模式的指标。

用法(与官方评测同一套环境变量, GPU 服务器上运行):
  DATASET=games MODEL=<checkpoint路径或stage1/stage2> \
      .venv/bin/python -m src.eval.sidreasoner_eval_short --mode all
  # --mode eval|think 只跑其一; --max-len N 调阈值(如 5)

产物:
  temp/sidreasoner/short/<Category>_le<N>.csv                     筛出的子集
  results/sidreasoner/<Category>_le<N>_{eval,think}_<模型名>.json 评测输出
"""

import argparse
import ast
import os
import subprocess
import sys

import pandas as pd

from config import load_config
from src.train.common import resolve_category


def build_subset(data_cfg, max_len):
    category = resolve_category(data_cfg)
    fname = f"{category}_5_2016-10-2018-11.csv"
    src = os.path.join(data_cfg["root"], "test", fname)
    df = pd.read_csv(src)
    lens = df["history_item_sid"].map(
        lambda s: len(ast.literal_eval(s)) if isinstance(s, str) else 999
    )
    sub = df[lens <= max_len]
    out_dir = os.path.join("temp", "sidreasoner", "short")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{category}_le{max_len}.csv")
    sub.to_csv(out, index=False)
    print(
        f"[short] {category}: {len(sub)}/{len(df)} 行 history<= {max_len} -> {out}"
    )
    return out, category, fname


def run(mode, test_path, category, fname, max_len):
    cfg = load_config("sidreasoner")
    if mode == "eval":
        from src.eval.sidreasoner_eval import main
    else:
        from src.eval.sidreasoner_eval_think import main

    stage = os.environ.get("MODEL") or cfg["evaluation"].get(
        "eval_model" if mode == "eval" else "think_eval_model", "stage1"
    )
    label = (
        stage
        if stage in ("stage1", "stage2")
        else os.path.basename(os.path.normpath(str(stage)))
    )
    result = os.path.join(
        "results",
        "sidreasoner",
        f"{category}_le{max_len}_{mode}_{label}.json",
    )
    main(test_data_path=test_path, result_json_data=result)

    from src.eval.sidreasoner_metrics import calc

    info_file = os.path.join(
        cfg["data"]["root"], "info", fname.rsplit(".", 1)[0] + ".txt"
    )
    print(f"\n[short:{mode}] 指标({result}):")
    calc(result, info_file)


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "think", "all"], default="all")
    ap.add_argument("--max-len", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config("sidreasoner")
    test_path, category, fname = build_subset(cfg["data"], args.max_len)
    if args.mode == "all":
        # 两种模式各起一个子进程: 保证 HF 模型/vLLM 的显存互不叠加
        for m in ("eval", "think"):
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "src.eval.sidreasoner_eval_short",
                    "--mode",
                    m,
                    "--max-len",
                    str(args.max_len),
                ]
            )
    else:
        run(args.mode, test_path, category, fname, args.max_len)


if __name__ == "__main__":
    main_cli()
