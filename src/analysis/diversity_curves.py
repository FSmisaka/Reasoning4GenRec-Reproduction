"""绘制多样性分析指标随 RL training step 变化的曲线。

用法：
  .venv/bin/python -m src.analysis.diversity_curves \
      results/diversity/Games/think_rolloutbeam
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

from src.eval.sidreasoner_reward_curves import C_BLUE, C_RED, SAVE_DPI, setup_style

import matplotlib.pyplot as plt


# 每个条目对应一条折线。key 按顺序列出 JSON 中的各层字段名。
DEFAULT_SERIES = (
    {"key": ("text", "duplication", "cosine", "mean"),
     "label": "Mean cosine similarity", "color": C_BLUE, "ls": "-", "marker": "o"},
    {"key": ("text", "duplication", "jaccard3", "mean"),
     "label": "Mean 3-gram Jaccard similarity", "color": C_RED, "ls": "--", "marker": "s"},
)

STEP_FILE = re.compile(r"step_(\d+)(?:_own)?\.json")


def collect_series(folder: Path) -> dict[tuple[str, ...], list[tuple[int, float]]]:
    """每个 step 文件只读取一次，按步数升序收集各指标的有效值。"""
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
        raise ValueError(f"{folder} 中没有 step_N.json 或 step_N_own.json")

    points = {spec["key"]: [] for spec in DEFAULT_SERIES}
    for step, path in sorted(files.items()):
        try:
            with path.open(encoding="utf-8") as stream:
                result = json.load(stream)
        except ValueError as exc:
            raise ValueError(f"{path} JSON 解析失败：{exc}") from exc

        for key, series_points in points.items():
            value = result
            for field in key:
                value = value.get(field) if isinstance(value, dict) else None
            if type(value) in (int, float) and math.isfinite(value):
                series_points.append((step, value))
            else:
                print(f"[warn] {path} 缺少有效的 {'.'.join(key)}，跳过该点", file=sys.stderr)
    return points


def plot_diversity_curves(folder: Path, out_path: Path | None = None) -> Path:
    """绘制 DEFAULT_SERIES 中配置的指标，并返回图片路径。"""
    folder = folder.resolve()
    points = collect_series(folder)
    steps = sorted({step for series in points.values() for step, _ in series})
    if not steps:
        raise ValueError(f"{folder} 中没有可绘制的指标数据")

    output = out_path or folder / "diversity_curves.png"
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context():
        setup_style()
        fig, ax = plt.subplots(figsize=(5.5, 2.8), constrained_layout=True)
        try:
            for spec in DEFAULT_SERIES:
                series = points[spec["key"]]
                if not series:
                    continue
                xs, ys = zip(*series)
                ax.plot(xs, ys, label=spec["label"], color=spec["color"],
                        ls=spec["ls"], marker=spec["marker"], ms=3.5, lw=1.3,
                        markeredgewidth=0.9, markerfacecolor="white")

            stride = max(1, math.ceil((len(steps) - 1) / 9))
            ax.set_xticks((*steps[:-1:stride], steps[-1]))
            ax.set_xlabel("RL training step")
            ax.set_ylabel("Mean pairwise similarity")
            ax.set_title(f"{folder.parent.name} · {folder.name}")
            ax.grid(True, ls=":", lw=0.6, alpha=0.6)
            ax.spines[["top", "right"]].set_visible(False)

            # constrained_layout 为图下方的图例预留空间。
            fig.legend(loc="outside lower center", ncol=min(4, len(ax.lines)),
                       frameon=False, columnspacing=1.4, handlelength=1.8)
            fig.savefig(output, dpi=SAVE_DPI, bbox_inches="tight")
        finally:
            plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="包含 step_N.json 的多样性结果目录")
    parser.add_argument("--out", type=Path, help="输出图片路径（默认写入输入目录的 diversity_curves.png）")
    args = parser.parse_args()
    try:
        output = plot_diversity_curves(args.folder, args.out)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
