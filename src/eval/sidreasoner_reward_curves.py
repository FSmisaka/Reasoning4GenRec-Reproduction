"""Stage 3 Reward 组成曲线: 逐 step 绘制正确性 / 多样性 / 总 reward 统计。

数据来源(训练时自动写入, 每 train step 一行 JSONL, 由 rl.sh 指定路径
logs/<Category>_stage3_rl_Qwen3-1.7B<后缀>_interdiv_metrics.jsonl):

    critic/correctness_rewards/{mean,max,min}   正确性 reward(StepRule / Rollout-Beam)
    critic/diversity_rewards/{mean,max,min}     多样性 reward(BLEU / n-gram)
    critic/total_rewards/{mean,max,min}         总 reward(= verl critic/rewards)

想展示哪些曲线, 改下方 DEFAULT_SERIES 即可(每行一条: label=图例,
exp=实验名后缀(_bleu/_ngram/...), key=JSONL 字段); 命令行 --metrics
可指定单个 JSONL 覆盖所有系列的 exp 解析。

用法(需 .venv 内有 matplotlib):
  DATASET=games .venv/bin/python -m src.eval.sidreasoner_reward_curves
  make games-sidreasoner-reward-curves                       # 等价
  make games-sidreasoner-reward-curves ARGS="--out /tmp/r.png"
"""

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager, rcParams  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")

from config import load_config  # noqa: E402

SAVE_DPI = 300
LOGS_ROOT = "logs"

# 配色与 sidreasoner_rl_curves.py 一致(蓝/红/灰主色 + 金色强调)
C_BLUE = "#3572C6"
C_RED = "#C94F4F"
C_GRAY = "#4B5563"
C_GOLD = "#F2B01E"

_LATIN_CANDIDATES = ("Times New Roman",)
_CJK_CANDIDATES = ("Songti SC", "SimSun", "STSong")

# 支持的 reward 统计键(校验 + 文档), 绘图取值时经 KEY_ALIASES 兼容旧字段
SUPPORTED_KEYS = tuple(
    f"critic/{g}_rewards/{s}"
    for g in ("correctness", "diversity", "total")
    for s in ("mean", "max", "min")
)
SUPPORTED_KEYS += ('niche/c1_mean', 'niche/c2_mean','niche/c3_mean')
SUPPORTED_KEYS += ('td_mean', 'td_success_mean', 'neighbor_sim_mean', 'valid_neighbors_mean')

KEY_ALIASES = {
    "critic/correctness_rewards/mean": "base_reward_mean",
    "critic/diversity_rewards/mean": "intrinsic_mean",
    "critic/total_rewards/mean": "augmented_reward_mean",
}

# 曲线注册表: 加新折线只需在此追加一行(label=图例, exp=实验名后缀,
# key=JSONL 统计键); 也可在调用 plot_reward_curves(series=...) 时整体替换。
# exp 会被拼进 logs/<C>_stage3_rl_Qwen3-1.7B*{exp}_interdiv_metrics.jsonl
# (实验后缀 = _<corr>_<div>, 如 _steprule_ngram / _rolloutbeam_bleu)
# 通配(兼容 _own 等 SID 后缀); 多个匹配取最新修改的文件。
DEFAULT_SERIES = [
    {"label": "niche1", "exp": "_rolloutbeam_niche",
     "key": "niche/c1_mean", "color": C_RED, "ls": "-", "marker": None},
    {"label": "niche2", "exp": "_rolloutbeam_niche",
     "key": "niche/c2_mean", "color": C_BLUE, "ls": "-", "marker": None},
    {"label": "niche3", "exp": "_rolloutbeam_niche",
     "key": "niche/c3_mean", "color": C_GRAY, "ls": "-", "marker": None},
]

'''
DEFAULT_SERIES = [
    {"label": "Correctness (Beam x n-gram)", "exp": "_rolloutbeam_ngram",
     "key": "critic/correctness_rewards/mean", "color": C_BLUE, "ls": ":", "marker": "v"},
    {"label": "Diversity (Beam x n-gram)", "exp": "_rolloutbeam_ngram",
     "key": "critic/diversity_rewards/mean", "color": C_RED, "ls": ":", "marker": "D"},
    {"label": "Total (Beam x n-gram)", "exp": "_rolloutbeam_ngram",
     "key": "critic/total_rewards/mean", "color": C_GRAY, "ls": "-.", "marker": "^"},

    {"label": "Correctness (Beam x llmjudge)", "exp": "_rolloutbeam_llmjudge",
     "key": "critic/correctness_rewards/mean", "color": C_BLUE, "ls": "-", "marker": None},
    {"label": "Diversity (Beam x llmjudge)", "exp": "_rolloutbeam_llmjudge",
     "key": "critic/diversity_rewards/mean", "color": C_RED, "ls": "-", "marker": None},
    {"label": "Total (Beam x llmjudge)", "exp": "_rolloutbeam_llmjudge",
     "key": "critic/total_rewards/mean", "color": C_GRAY, "ls": "-", "marker": None},

    {"label": "Correctness (Beam x niche)", "exp": "_rolloutbeam_niche",
     "key": "critic/correctness_rewards/mean", "color": C_BLUE, "ls": "-", "marker": None},
    {"label": "Diversity (Beam x niche)", "exp": "_rolloutbeam_niche",
     "key": "critic/diversity_rewards/mean", "color": C_RED, "ls": "-", "marker": None},
    {"label": "Total (Beam x niche)", "exp": "_rolloutbeam_niche",
     "key": "critic/total_rewards/mean", "color": C_GRAY, "ls": "-", "marker": None},
]
'''

def setup_style():
    """印刷尺寸优先的样式(与 sidreasoner_rl_curves.py 完全一致)。"""
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


def resolve_metrics_path(category: str, exp: str, logs_root: str = LOGS_ROOT):
    """实验后缀 -> interdiv 指标 JSONL(通配 _own 等 SID 后缀, 取最新)。"""
    pattern = os.path.join(
        logs_root, f"{category}_stage3_rl_Qwen3-1.7B*{exp}_interdiv_metrics.jsonl"
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[warn] {pattern} 匹配 {len(matches)} 个文件, 取最新: {matches[-1]}")
    return matches[-1]


def load_metrics(path: str) -> list:
    """读 JSONL -> 按 step 升序的 record 列表(坏行跳过并告警)。"""
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] {path}:{i + 1} 非法 JSON, 跳过")
                continue
            if "step" in rec:
                records.append(rec)
    records.sort(key=lambda r: int(r["step"]))
    return records


def series_points(records: list, key: str) -> list:
    """一条折线的 (step, value) 序列(缺 step / 缺键的行自然跳过)。

    新键缺失时按 KEY_ALIASES 回退旧字段(base_reward_mean 等, 兼容
    本改动之前产生的 JSONL; 旧字段无 max/min, 折线只含 mean)。
    """
    if key not in SUPPORTED_KEYS:
        raise ValueError(f"未知统计键 {key!r}, 可选: {SUPPORTED_KEYS}")
    alias = KEY_ALIASES.get(key)
    pts = []
    for rec in records:
        v = rec.get(key)
        if v is None and alias is not None:
            v = rec.get(alias)
        if v is not None:
            pts.append((int(rec["step"]), float(v)))
    return sorted(pts)


def plot_reward_curves(
    dataset: str = "games",
    series: list = None,
    out_path: str = None,
    logs_root: str = LOGS_ROOT,
    metrics_path: str = None,
    results_root: str = "results/sidreasoner",
    figsize=(5.5, 2.8),
    xtick_every: int = 200,
):
    """画一张"reward 统计 vs RL step"的多曲线图, 返回保存路径。

    Args:
        dataset: games / office / industrial(决定实验名里的 Category)。
        series: 折线配置(默认 DEFAULT_SERIES), 元素含 label / exp / key,
            可选 color / ls / marker / metrics(直接指定 JSONL 路径)。
        out_path: 输出 png(默认 results/sidreasoner/<D>/figures/rewards.png)。
        logs_root / metrics_path: JSONL 所在目录 / 显式单文件(覆盖 exp 解析)。
        figsize / xtick_every: 图尺寸与横轴刻度间隔。
    """
    category = load_config("sidreasoner")["data"]["categories"][dataset]
    series = [dict(s) for s in (series if series is not None else DEFAULT_SERIES)]

    setup_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    all_steps = []
    plotted = 0
    for s in series:
        path = s.get("metrics") or metrics_path or resolve_metrics_path(
            category, s["exp"], logs_root
        )
        if not path or not os.path.exists(path):
            print(f"[warn] 找不到指标文件(exp={s.get('exp')}): {path}, 跳过该折线")
            continue
        pts = series_points(load_metrics(path), s["key"])
        if not pts:
            print(f"[warn] {path}: 无 {s['key']} 可用数据, 跳过该折线")
            continue
        steps = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        all_steps.extend(steps)
        ax.plot(steps, vals, color=s.get("color", C_GOLD), ls=s.get("ls", "-"),
                marker=s.get("marker", "o"), ms=3.5, lw=1.3,
                markeredgewidth=0.9, markerfacecolor="white", label=s["label"])
        print(f"[{os.path.basename(path)}] {s['key']}: {len(pts)} 个点, "
              f"step {steps[0]}-{steps[-1]}, {min(vals):.4f}~{max(vals):.4f}")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise SystemExit("没有任何折线有数据, 不出图(先跑 rl-bleu/rl-ngram 训练)")

    if all_steps:
        lo = (min(all_steps) // 100) * 100
        hi = ((max(all_steps) + 99) // 100) * 100
        ax.set_xticks(range(lo, hi + 1, xtick_every))
    ax.set_xlabel("RL training step")
    ax.set_ylabel("Reward")
    ax.set_title(f"{category} · Stage 3 Reward")
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # 图例置于图片下方(与 sidreasoner_rl_curves.py 一致)
    fig.legend(loc="outside lower center",
               ncol=3 if plotted >= 5 else min(4, plotted),
               frameon=False, columnspacing=1.4, handlelength=1.8)

    if out_path is None:
        from src.eval.result_routing import DATASET_DIRS

        out_path = os.path.join(
            results_root, DATASET_DIRS[category], "figures", "rewards.png"
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=os.environ.get("DATASET", "games"),
                    choices=("games", "office", "industrial"))
    ap.add_argument("--metrics", default=None,
                    help="显式指定 JSONL(覆盖所有系列的 exp 解析)")
    ap.add_argument("--out", default=None, help="输出 png 路径")
    ap.add_argument("--logs-root", default=LOGS_ROOT, help="指标 JSONL 所在目录")
    args = ap.parse_args()

    plot_reward_curves(
        dataset=args.dataset,
        out_path=args.out,
        logs_root=args.logs_root,
        metrics_path=args.metrics,
    )


if __name__ == "__main__":
    main()
