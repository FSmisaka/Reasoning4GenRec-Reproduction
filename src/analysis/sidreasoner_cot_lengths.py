"""统计思考模式评测中每个 checkpoint 的平均 CoT token 数。"""

import argparse
import json
import re
from math import ceil
from pathlib import Path
from statistics import mean

from src.eval.sidreasoner_reward_curves import C_RED, SAVE_DPI, setup_style

import matplotlib.pyplot as plt


STEP_FILE = re.compile(r"step_(\d+)(?:_own)?\.json")
OUTPUT_DIR = Path("results/length")


def mean_cot_lengths(folder: Path) -> dict[int, float]:
    """按 step 读取逐样本 cot_len，每条测试记录计一次。"""
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

    averages = {}
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
        for index, record in enumerate(records, start=1):
            length = record.get("cot_len") if isinstance(record, dict) else None
            if type(length) is not int or length < 0:
                raise ValueError(f"{path} 的第 {index} 条记录缺少有效的 cot_len")
            lengths.append(length)
        averages[step] = float(mean(lengths))
    return averages


def plot_mean_lengths(averages: dict[int, float], title: str, output_path: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 2.8), constrained_layout=True)
    steps, lengths = zip(*sorted(averages.items()))
    ax.plot(
        steps, lengths, color=C_RED, ls="--", marker="s", ms=3.5, lw=1.3,
        markeredgewidth=0.9, markerfacecolor="white", label="Mean CoT length",
    )
    stride = max(1, ceil((len(steps) - 1) / 9))
    ax.set_xticks((*steps[:-1:stride], steps[-1]))
    ax.set_xlabel("RL training step")
    ax.set_ylabel("CoT length (tokens)")
    ax.set_title(title)
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.legend(loc="outside lower center", frameon=False, handlelength=1.8)
    try:
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="包含 step_N.json 的评测结果目录")
    args = parser.parse_args()

    folder = args.folder.resolve()
    try:
        averages = mean_cot_lengths(folder)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{folder.parent.name}_{folder.name}_cot_length"
        json_path = OUTPUT_DIR / f"{stem}.json"
        png_path = OUTPUT_DIR / f"{stem}.png"
        plot_mean_lengths(averages, f"{folder.parent.name} · Stage 3 RL", png_path)
        json_path.write_text(json.dumps(averages, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")

    print(f"[saved] {json_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
