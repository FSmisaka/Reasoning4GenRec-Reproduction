"""Stage 3 (RL) 训练曲线: 按 step 对比不同 reward 变体在思考模式评测下的指标。

输入一个指标名(如 NDCG@10), 输出一张 3 条折线的对比图, 保存到
results/sidreasoner/<Dataset>/figures/<指标>.png。

数据来源(与评测产物结构一一对应, 无需额外记录):
- think_<variant>/step_N.json          贪心评测 -> 逐样本记录, 现场重算指标
- think_<variant>_temp/step_N_summary  采样评测 -> aggregate.metrics[指标].mean
  (同时画 std 置信带)

指标可选: NDCG@5 / NDCG@10 / HR@5 / HR@10(Recall/HitRate 均识别为 HR;
重算走 sidreasoner_metrics.compute_metrics, K 需不超过评测束宽 10)。

用法(服务器, 需 .venv 内有 matplotlib):
  DATASET=games .venv/bin/python -m src.eval.sidreasoner_rl_curves --metric NDCG@10
  DATASET=games .venv/bin/python -m src.eval.sidreasoner_rl_curves --all   # 4 张全画
或经 make: make games-sidreasoner-curves              # 默认 --all
          make games-sidreasoner-curves ARGS="--metric HR@10"
"""

import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, rcParams  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")

from config import load_config  # noqa: E402
from src.eval.result_routing import DATASET_DIRS, RESULT_ROOT  # noqa: E402
from src.eval.sidreasoner_metrics import compute_metrics, load_item_names  # noqa: E402

SAVE_DPI = 300

# 配色与参考绘图脚本一致(蓝/红/灰主色 + 金色强调)
C_BLUE = "#3572C6"
C_RED = "#C94F4F"
C_GRAY = "#4B5563"
C_GOLD = "#F2B01E"

_LATIN_CANDIDATES = ("Times New Roman",)
_CJK_CANDIDATES = ("Songti SC", "SimSun", "STSong")

# 支持的指标(别名归一: Recall*/HitRate* -> HR*)
METRIC_ALIASES = {
    "ndcg": "NDCG",
    "hr": "HR",
    "hitrate": "HR",
    "recall": "HR",
    "recallhitrate": "HR",
    "hitraterecall": "HR",
}
SUPPORTED_METRICS = ("NDCG@5", "NDCG@10", "HR@5", "HR@10")

# 曲线注册表: 以后加新折线只需在此追加一行(label=图例, subdir=结果子目录)。
# 也可在调用 plot_rl_metric(series=...) 时整体替换/增删。
DEFAULT_SERIES = [
    {"label": "Rollout-Beam (OneRec-Think)",
     "subdir": "think_rolloutbeam", "color": C_BLUE, "ls": "-", "marker": "o"},
    {"label": "Rollout-Beam x R_div (OneReason)",
     "subdir": "think_rolloutbeam_div", "color": C_RED, "ls": "--", "marker": "s"},
    {"label": "StepRule (原始 SIDReasoner)",
     "subdir": "think_steprule", "color": C_GRAY, "ls": "-.", "marker": "^"},
]


def setup_style():
    """印刷尺寸优先的样式(与仓库统一绘图规范一致)。"""
    avail = {f.name for f in font_manager.fontManager.ttflist}
    latin = [f for f in _LATIN_CANDIDATES if f in avail]
    cjk = [f for f in _CJK_CANDIDATES if f in avail]
    rcParams["font.family"] = latin + cjk + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["mathtext.fontset"] = "stix"
    rcParams.update(
        {
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.edgecolor": "#8A8F98",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#444950",
            "ytick.color": "#444950",
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def normalize_metric(metric: str) -> str:
    """把 NDCG@10 / Recall@10 / HitRate(HR)@5 等别名归一为 NDCG@K / HR@K。"""
    m = re.match(r"^([A-Za-z()\s]+)@(\d+)$", str(metric).strip())
    if not m:
        raise ValueError(f"指标名格式应为 <名称>@<K>, 收到: {metric!r}")
    name = re.sub(r"[^a-z]", "", m.group(1).lower())
    if name not in METRIC_ALIASES:
        raise ValueError(f"未知指标 {metric!r}, 可选: {SUPPORTED_METRICS}")
    return f"{METRIC_ALIASES[name]}@{int(m.group(2))}"


def metric_ylabel(metric_key: str) -> str:
    return f"{metric_key} (Recall)" if metric_key.startswith("HR@") else metric_key


def _step_files(subdir_path: str):
    """扫描目录, 返回 {step: (greedy_path, summary_path)}(_own 后缀兼容)。"""
    found = {}
    if not os.path.isdir(subdir_path):
        return found
    for fname in os.listdir(subdir_path):
        m = re.match(r"^step_(\d+)(?:_own)?\.json$", fname)
        if m:
            step = int(m.group(1))
            found.setdefault(step, [None, None])[0] = os.path.join(subdir_path, fname)
            continue
        m = re.match(r"^step_(\d+)(?:_own)?_summary\.json$", fname)
        if m:
            step = int(m.group(1))
            found.setdefault(step, [None, None])[1] = os.path.join(subdir_path, fname)
    return found


def series_points(subdir_path, item_names, metric_key):
    """一条折线的 (step, value[, std]) 序列, 按 step 升序。

    优先用采样评测的 summary(mean/std), 否则对贪心评测的逐样本记录
    现场调用 compute_metrics 重算。缺 step 的折线自然到此截断。
    """
    pts = []
    for step, (greedy, summary) in sorted(_step_files(subdir_path).items()):
        if summary is not None:
            with open(summary, "r") as f:
                agg = json.load(f)["aggregate"]["metrics"]
            if metric_key not in agg:
                print(f"[warn] {summary}: 无指标 {metric_key}, 跳过")
                continue
            pts.append((step, float(agg[metric_key]["mean"]),
                        float(agg[metric_key].get("std", 0.0))))
        else:
            with open(greedy, "r") as f:
                test_data = json.load(f)
            mets = compute_metrics(test_data, item_names)
            if metric_key not in mets:
                print(f"[warn] {greedy}: 无指标 {metric_key}(束宽不足?), 跳过")
                continue
            pts.append((step, float(mets[metric_key]), None))
    return pts


def plot_rl_metric(
    metric,
    dataset="games",
    series=None,
    out_path=None,
    root=None,
    figsize=(5.5, 2.8),
    xtick_every=200,
):
    """画一张"指标 vs RL step"的多变体对比折线图, 返回保存路径。

    Args:
        metric: 指标名(别名见 normalize_metric), 如 NDCG@10 / Recall@5。
        dataset: games / office / industrial。
        series: 折线配置列表(默认 DEFAULT_SERIES), 元素含
            label / subdir / color / ls / marker, 可只给 label+subdir。
        out_path: 输出路径(默认 results/sidreasoner/<D>/figures/<指标>.png)。
        root: 结果根目录(默认 results/sidreasoner, 供测试注入)。
        figsize / xtick_every: 图尺寸与横轴刻度间隔。
    """
    cfg = load_config("sidreasoner")
    category = cfg["data"]["categories"][dataset]
    ds_dir = DATASET_DIRS[category]
    root = root or RESULT_ROOT
    info_file = os.path.join(
        cfg["data"]["root"], "info", f"{category}_5_2016-10-2018-11.txt"
    )
    item_names = load_item_names(info_file)

    metric_key = normalize_metric(metric)
    series = [dict(s) for s in (series if series is not None else DEFAULT_SERIES)]

    setup_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    all_steps = []
    plotted = 0
    for s in series:
        pts = series_points(os.path.join(root, ds_dir, s["subdir"]),
                            item_names, metric_key)
        if not pts:
            print(f"[warn] {ds_dir}/{s['subdir']}: 没有 {metric_key} 可用的 step 文件, 跳过该折线")
            continue
        steps = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        stds = [p[2] for p in pts]
        all_steps.extend(steps)
        ax.plot(steps, vals, color=s.get("color", C_GOLD), ls=s.get("ls", "-"),
                marker=s.get("marker", "o"), ms=3.5, lw=1.3,
                markeredgewidth=0.9, markerfacecolor="white",
                label=s["label"])
        # 采样评测(summary)带 std 时画置信带
        if any(st and st > 0 for st in stds):
            ax.fill_between(steps,
                            [v - st for v, st in zip(vals, stds)],
                            [v + st for v, st in zip(vals, stds)],
                            color=s.get("color", C_GOLD), alpha=0.13, lw=0)
        print(f"[{ds_dir}/{s['subdir']}] {len(pts)} 个点: "
              f"step {steps[0]}-{steps[-1]}, {metric_key} "
              f"{min(vals):.4f}~{max(vals):.4f}")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise SystemExit(f"没有任何折线有数据({metric_key}), 不出图")

    if all_steps:
        lo = (min(all_steps) // 100) * 100
        hi = ((max(all_steps) + 99) // 100) * 100
        ax.set_xticks(range(lo, hi + 1, xtick_every))
    ax.set_xlabel("RL training step")
    ax.set_ylabel(metric_ylabel(metric_key))
    ax.set_title(f"{ds_dir} · Stage 3 RL")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # 图例置于图片下方(outside lower center + constrained_layout, 不与内容重叠)
    fig.legend(loc="outside lower center", ncol=min(3, plotted),
               frameon=False, columnspacing=1.4, handlelength=1.8)

    if out_path is None:
        out_path = os.path.join(root, ds_dir, "figures", f"{metric_key}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metric", default=None,
                    help="指标名, 如 NDCG@10 / HR@5(Recall/HitRate 等别名均可)")
    ap.add_argument("--all", action="store_true",
                    help=f"一次画全部 {len(SUPPORTED_METRICS)} 张: {', '.join(SUPPORTED_METRICS)}")
    ap.add_argument("--dataset", default=os.environ.get("DATASET", "games"),
                    choices=("games", "office", "industrial"))
    args = ap.parse_args()

    if args.all:
        metrics = list(SUPPORTED_METRICS)
    elif args.metric:
        metrics = [normalize_metric(args.metric)]
    else:
        ap.error("请给 --metric <指标> 或 --all")

    for m in metrics:
        plot_rl_metric(m, dataset=args.dataset)


if __name__ == "__main__":
    main()
