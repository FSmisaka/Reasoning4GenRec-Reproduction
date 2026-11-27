"""统计思考模式评测中每个 checkpoint 的平均 CoT 长度和未闭合率。"""

import argparse
import json
import re
from math import ceil
from pathlib import Path
from statistics import mean

from src.eval.sidreasoner_reward_curves import C_BLUE, C_RED, SAVE_DPI, setup_style

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# 默认绘制注册表中所有曲线。--series 可按 key 选择其中一条或多条。
DEFAULT_SERIES = [
    {"key": "length", "index": 0, "label": "Mean CoT length",
     "ylabel": "CoT length (tokens)", "color": C_RED, "ls": "--", "marker": "s"},
    {"key": "unclosed_rate", "index": 1, "label": "Unclosed CoT rate",
     "ylabel": "Unclosed CoT rate (%)", "color": C_BLUE, "ls": "-", "marker": "o",
     "percent": True},
]

STEP_FILE = re.compile(r"step_(\d+)(?:_own)?\.json")
OUTPUT_DIR = Path("results/length")


def collect_cot_stats(folder: Path) -> dict[int, tuple[float, float]]:
    """逐 step 返回（平均 token 数，close_found=False 的比例）。"""
    files = {}
    for path in sorted(folder.iterdir()):
        match = STEP_FILE.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        step = int(match[1])
        if step in files:
            raise ValueError(f"step {step} 有两份结果文件：{files[step]} 和 {path}")
        files[step] = path
    if not files:
        raise ValueError(f"{folder} 没有找到 step_N.json 评测结果")

    stats = {}
    for step, path in sorted(files.items()):
        try:
            with path.open(encoding="utf-8") as stream:
                records = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} JSON 解析失败：{exc}") from exc
        if isinstance(records, dict):
            records = records.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path} 没有逐样本评测记录")

        lengths = []
        unclosed = 0
        for index, record in enumerate(records, start=1):
            length = record.get("cot_len") if isinstance(record, dict) else None
            if type(length) is not int or length < 0:
                raise ValueError(f"{path} 的第 {index} 条记录缺少有效的 cot_len")
            closed = record.get("close_found")
            if type(closed) is not bool:
                raise ValueError(f"{path} 的第 {index} 条记录缺少布尔值 close_found")
            lengths.append(length)
            unclosed += not closed
        stats[step] = (float(mean(lengths)), unclosed / len(records))
    return stats


def plot_cot_stats(stats, series, title: str, output_path: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 2.8), constrained_layout=True)
    steps, values = zip(*sorted(stats.items()))
    for index, spec in enumerate(series):
        axis = ax if index == 0 else ax.twinx()
        axis.plot(
            steps, [row[spec["index"]] for row in values], color=spec["color"],
            ls=spec["ls"], marker=spec["marker"], ms=3.5, lw=1.3,
            markeredgewidth=0.9, markerfacecolor="white", clip_on=False, label=spec["label"],
        )
        axis.set_ylabel(spec["ylabel"])
        axis.spines["top"].set_visible(False)
        if spec.get("percent"):
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            lo, hi = axis.get_ylim()
            axis.set_ylim(max(0, lo), min(1, hi))

    stride = max(1, ceil((len(steps) - 1) / 9))
    ax.set_xticks((*steps[:-1:stride], steps[-1]))
    ax.set_xlabel("RL training step")
    ax.set_title(title)
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.spines["right"].set_visible(False)

    fig.legend(loc="outside lower center", ncol=len(series), frameon=False, handlelength=1.8)
    try:
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="包含 step_N.json 的评测结果目录")
    parser.add_argument(
        "--series", nargs="+", choices=[spec["key"] for spec in DEFAULT_SERIES],
        help="选择要绘制的曲线，默认绘制 DEFAULT_SERIES 中的全部曲线",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    series = [
        spec for spec in DEFAULT_SERIES
        if args.series is None or spec["key"] in args.series
    ]
    try:
        stats = collect_cot_stats(folder)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{folder.parent.name}_{folder.name}_cot_length"
        json_path = OUTPUT_DIR / f"{stem}.json"
        png_path = OUTPUT_DIR / f"{stem}.png"
        plot_cot_stats(stats, series, f"{folder.parent.name} · Stage 3 RL", png_path)
        json_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    print(f"[saved] {json_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
