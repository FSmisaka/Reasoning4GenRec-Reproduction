"""
SIDReasoner 评估结果后处理

split:     把测试 CSV 均分给多张卡(迁移自 split.py)
merge:     合并多卡拆分评估产生的分片 json(迁移自 merge.py)
calc:      计算 HR@K / NDCG@K(迁移自 calc.py, K 取 1/3/5/10/20/50
           中不超过束宽的值; 匹配方式为 SID 串全等, 同名多物品共享排名);
           输入若是多 seed 评测的 *_summary.json 则直接打印聚合结果
aggregate: 对同一模型按多个随机种子评测得到的多份 result json,
           计算每个指标的 均值 / 样本标准差 / 95% 置信区间

用法:
  DATASET=games .venv/bin/python -m src.eval.sidreasoner_metrics \
      calc --path <result.json>
  DATASET=games .venv/bin/python -m src.eval.sidreasoner_metrics \
      aggregate --path <result_seed42.json>,<result_seed123.json>,...
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

TOPK_LIST = [1, 3, 5, 10, 20, 50]

# 双测 t 分位数表 {置信水平: {自由度: t 值}}, df 1..30;
# scipy 可用时优先用 scipy.stats.t.ppf 精确计算。
_T_TABLE = {
    0.90: {
        1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943,
        7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 15: 1.753, 20: 1.725,
        25: 1.708, 30: 1.697,
    },
    0.95: {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
        25: 2.060, 30: 2.042,
    },
    0.99: {
        1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707,
        7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169, 15: 2.947, 20: 2.845,
        25: 2.787, 30: 2.750,
    },
}


def split(input_path, output_path, cuda_list, num_samples=-1):
    if isinstance(cuda_list, int):
        cuda_list = [cuda_list]
    df = pd.read_csv(input_path)
    if num_samples is not None and int(num_samples) > -1:
        num_samples = int(num_samples)
        df = df.head(num_samples) if num_samples > 0 else df.iloc[0:0]
    os.makedirs(output_path, exist_ok=True)
    df_len = len(df)
    cuda_list = list(cuda_list)
    for i in range(len(cuda_list)):
        start = i * df_len // len(cuda_list)
        end = (i + 1) * df_len // len(cuda_list)
        df[start:end].to_csv(
            f"{output_path}/{cuda_list[i]}.csv", index=True
        )


def merge(input_path, output_path, cuda_list):
    if isinstance(cuda_list, int):
        cuda_list = [cuda_list]
    cuda_list = list(cuda_list)
    result = []
    for gpu in cuda_list:
        shard = os.path.join(input_path, f"{gpu}.json")
        with open(shard, "r") as f:
            result.extend(json.load(f))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"[merge] {len(result)} samples -> {output_path}")


def load_item_names(item_path):
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    with open(f"{item_path}.txt", "r") as f:
        items = f.readlines()
    return [_.split("\t")[0].strip() for _ in items]


def compute_metrics(test_data, item_names):
    """对一份 think 评测结果(list[dict], 含 predict/output 字段)计算指标。

    返回 {"NDCG@K": v, "HR@K": v}(K 取 TOPK_LIST 中不超过束宽的值),
    以及 "n_beam" / "n_test" / "invalid_predictions" 元信息。
    """
    item_dict = {}
    for i, name in enumerate(item_names):
        item_dict.setdefault(name, []).append(i)

    n_beam = None
    ALLNDCG = np.zeros(len(TOPK_LIST))
    ALLHR = np.zeros(len(TOPK_LIST))
    CC = 0

    text = [
        [_.strip('"\n').strip() for _ in sample["predict"]]
        for sample in test_data
    ]
    for index, sample in tqdm(enumerate(text), total=len(text)):
        if n_beam is None:
            n_beam = len(sample)
        if type(test_data[index]["output"]) == list:
            target_item = (
                test_data[index]["output"][0].strip('"').strip(" ")
            )
        else:
            target_item = test_data[index]["output"].strip(' \n"')
        minID = 1000000
        for i in range(len(sample)):
            if sample[i] not in item_dict:
                CC += 1
            if sample[i] == target_item:
                minID = i
                break
        for index, topk in enumerate(TOPK_LIST):
            if topk > n_beam:
                continue
            if minID < topk:
                ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                ALLHR[index] = ALLHR[index] + 1

    metrics = {}
    for index, topk in enumerate(TOPK_LIST):
        if topk > n_beam:
            continue
        metrics[f"NDCG@{topk}"] = float(
            ALLNDCG[index] / len(text) / (1.0 / math.log(2))
        )
        metrics[f"HR@{topk}"] = float(ALLHR[index] / len(text))
    metrics["n_beam"] = int(n_beam)
    metrics["n_test"] = len(text)
    metrics["invalid_predictions"] = int(CC)
    return metrics


def _t_critical(confidence, df):
    """双测 t 分位数: 优先 scipy, 缺失时用内置表(df<=30, 90/95/99%)。"""
    try:
        from scipy import stats

        return float(stats.t.ppf(0.5 + confidence / 2.0, df))
    except Exception:
        pass
    table = _T_TABLE.get(round(confidence, 2))
    if table is None:
        raise SystemExit(
            f"无 scipy 且置信水平 {confidence} 不在内置 t 表"
            f"({sorted(_T_TABLE)})支持范围"
        )
    if df in table:
        return table[df]
    if df > 30:
        return table[30]
    raise SystemExit(f"内置 t 表缺自由度 {df}")


def aggregate_metrics(per_seed_metrics, ci_level=0.95):
    """跨 seed 聚合: 每个指标报告 mean / std(样本标准差) / CI。

    per_seed_metrics: {seed: metrics_dict}(来自 compute_metrics)。
    CI = mean ± t_{ci_level, n-1} * std / sqrt(n)。
    """
    seeds = sorted(per_seed_metrics.keys())
    keys = [
        k
        for k in next(iter(per_seed_metrics.values())).keys()
        if not k.startswith(("n_beam", "n_test", "invalid"))
    ]
    n = len(seeds)
    agg = {}
    for key in keys:
        values = np.array(
            [per_seed_metrics[s][key] for s in seeds], dtype=float
        )
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        half = (
            _t_critical(ci_level, n - 1) * std / math.sqrt(n)
            if n > 1
            else 0.0
        )
        agg[key] = {
            "values": {s: per_seed_metrics[s][key] for s in seeds},
            "mean": mean,
            "std": std,
            "ci95_lower": mean - half,
            "ci95_upper": mean + half,
            "ci_half_width": half,
        }
    return {
        "seeds": seeds,
        "n_seeds": n,
        "ci_level": ci_level,
        "metrics": agg,
    }


def print_aggregate(agg, title="aggregate"):
    print(f"\n===== {title} (n={agg['n_seeds']}, "
          f"CI={agg['ci_level']:.0%}) =====")
    header = f"{'metric':<12}{'mean':>10}{'std':>10}{'ci95_lower':>12}" \
             f"{'ci95_upper':>12}"
    print(header)
    for key, m in agg["metrics"].items():
        print(
            f"{key:<12}{m['mean']:>10.6f}{m['std']:>10.6f}"
            f"{m['ci95_lower']:>12.6f}{m['ci95_upper']:>12.6f}"
        )


def calc(path, item_path):
    if isinstance(path, str):
        path = [path]
    item_names = load_item_names(item_path)

    for p in path:
        with open(p, "r") as f:
            test_data = json.load(f)

        # 多 seed 评测的汇总文件: 直接打印聚合结果
        if isinstance(test_data, dict) and "aggregate" in test_data:
            print(f"[calc] {p}: multi-seed summary")
            print_aggregate(
                test_data["aggregate"],
                title=f"{os.path.basename(p)}",
            )
            continue

        metrics = compute_metrics(test_data, item_names)
        print(metrics["n_beam"])
        valid_topk = [k for k in TOPK_LIST if k <= metrics["n_beam"]]
        print(valid_topk)
        ndcg = {k: v for k, v in metrics.items() if k.startswith("NDCG@")}
        hr = {k: v for k, v in metrics.items() if k.startswith("HR@")}
        print(f"NDCG:\t{[round(v, 6) for v in ndcg.values()]}")
        print(f"HR\t{[round(v, 6) for v in hr.values()]}")
        print(f"invalid predictions: {metrics['invalid_predictions']}")


def aggregate_cli(paths, item_path, ci_level):
    """对多份单 seed 结果 json 跨 seed 聚合。"""
    if isinstance(paths, str):
        paths = [p for p in paths.split(",") if p]
    item_names = load_item_names(item_path)
    per_seed = {}
    for p in paths:
        with open(p, "r") as f:
            test_data = json.load(f)
        seed = None
        if isinstance(test_data, dict) and "seed" in test_data:
            seed = test_data["seed"]
            test_data = test_data["samples"]
        else:
            stem = os.path.splitext(os.path.basename(p))[0]
            if "_seed" in stem:
                try:
                    seed = int(stem.rsplit("_seed", 1)[1])
                except ValueError:
                    pass
        if seed is None:
            raise SystemExit(
                f"无法从 {p} 解析 seed(文件名需含 _seed<数字> 或内容含 "
                "顶层 'seed' 字段); 也可改用 calc 逐份查看"
            )
        per_seed[seed] = compute_metrics(test_data, item_names)
        print(f"[aggregate] seed={seed}: {p}")
    agg = aggregate_metrics(per_seed, ci_level=ci_level)
    print_aggregate(agg, title=" | ".join(paths))
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["split", "merge", "calc", "aggregate"])
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--cuda_list", default="0")
    parser.add_argument("--path", default=None)
    parser.add_argument("--item_path", default=None)
    parser.add_argument(
        "--ci_level", type=float, default=None,
        help="置信区间水平(aggregate 模式, 默认取 config 的 evaluation.ci_level)",
    )
    args = parser.parse_args()

    if args.mode == "split":
        split(
            args.input_path,
            args.output_path,
            [int(x) for x in args.cuda_list.split(",")],
        )
    elif args.mode == "merge":
        merge(
            args.input_path,
            args.output_path,
            [int(x) for x in args.cuda_list.split(",")],
        )
    else:
        from config import load_config
        from src.train.common import resolve_category

        cfg = load_config("sidreasoner")
        category = resolve_category(cfg["data"])
        if args.path is None:
            raise SystemExit("--path is required for calc/aggregate")
        item_path = args.item_path or os.path.join(
            cfg["data"]["root"],
            "info",
            f"{category}_5_2016-10-2018-11.txt",
        )
        if args.mode == "calc":
            calc(args.path, item_path)
        else:
            aggregate_cli(
                args.path, item_path,
                ci_level=args.ci_level
                or cfg["evaluation"].get("ci_level", 0.95),
            )
