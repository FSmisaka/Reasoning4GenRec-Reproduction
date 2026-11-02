"""
思维链多样性分析 · 附属工具: 首句重复来源分解(同用户 vs 跨用户)

回答的问题: 语料中重复的首句(模板归并后), 究竟有多少来自"同一用户的
不同测试行"(级联结构可解释的部分), 有多少来自"不同用户"之间(真实的
跨用户模板化)。对贪心(T=0)与采样(T=0.6)语料各跑一遍, 即可对比解码
温度对该分解的影响。

两种归属口径(结论一致时可信):
  A 行归属: 每个重复组把 1 篇视为"原始", 组内出现最多的用户的多余行
    计为"同用户", 其余成员计为"跨用户";
  B 配对归属: 组内两两配对, 按两篇是否同一用户分类。

注意: 经 --import 导入的 think JSON 语料没有真实 user_id(以 #行号
代替), 每行 id 唯一, "同用户"恒为 0 —— 这类语料无法检验本假设,
脚本会给出警告。适用语料: collect_generate 生成的(如 stage2_temp/
stage3_temp, 含真实 user_id)。

用法:
  DATASET=games .venv/bin/python -m src.analysis.dup_decomposition \
      --corpora stage2_temp,stage3_temp [--top 8] [--json]
"""

import argparse
import itertools
import json
import os
from collections import Counter

from src.analysis.diversity_report import opener_template


def decompose(records):
    openers = [opener_template(r["cot"]) for r in records]
    users = [str(r.get("user_id", "")) for r in records]
    groups = {}
    for i, o in enumerate(openers):
        groups.setdefault(o, []).append(i)

    same_rows, cross_rows = 0, 0
    same_pairs, cross_pairs = 0, 0
    details = []
    for o, idxs in groups.items():
        if len(idxs) < 2:
            continue
        us = [users[i] for i in idxs]
        mu = Counter(us).most_common(1)[0][1]
        same_rows += mu - 1 if mu > 1 else 0
        cross_rows += len(idxs) - max(mu, 1)
        for i, j in itertools.combinations(idxs, 2):
            if users[i] == users[j]:
                same_pairs += 1
            else:
                cross_pairs += 1
        details.append(
            {"size": len(idxs), "users": us, "opener": o[:80]}
        )
    n = len(records)
    n_dup_members = sum(d["size"] for d in details)
    return {
        "n_docs": n,
        "n_dup_groups": len(details),
        "n_dup_members": n_dup_members,
        "dup_member_rate": round(n_dup_members / max(n, 1), 4),
        "redundant_rows": {
            "same_user": same_rows,
            "cross_user": cross_rows,
            "same_user_frac": round(
                same_rows / max(same_rows + cross_rows, 1), 4
            ),
        },
        "pairs": {
            "same_user": same_pairs,
            "cross_user": cross_pairs,
            "same_user_frac": round(
                same_pairs / max(same_pairs + cross_pairs, 1), 4
            ),
        },
        "top_groups": sorted(
            details, key=lambda d: -d["size"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="首句重复来源分解(详见模块 docstring)"
    )
    parser.add_argument(
        "--corpora", default=os.environ.get("CORPORA", ""),
        help="逗号分隔语料名(默认 corpora/ 下全部)",
    )
    parser.add_argument("--top", type=int,
                        default=int(os.environ.get("TOP", 8)),
                        help="展示的重复组明细条数")
    parser.add_argument("--json", action="store_true",
                        default=bool(os.environ.get("JSON")),
                        help="同时把结果写入 dup_decomposition.json")
    args = parser.parse_args()

    from config import load_config
    from src.train.common import resolve_category

    cfg = load_config("sidreasoner")
    category = resolve_category(cfg["data"])
    corpora_dir = os.path.join(
        "results", "diversity", category, "corpora"
    )
    if not os.path.isdir(corpora_dir):
        raise SystemExit(f"未找到 {corpora_dir}, 先运行采集")
    names = (
        [n.strip() for n in args.corpora.split(",") if n.strip()]
        if args.corpora
        else sorted(
            f[:-6] for f in os.listdir(corpora_dir)
            if f.endswith(".jsonl")
        )
    )

    results = {}
    for name in names:
        path = os.path.join(corpora_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"[decomp] 跳过不存在的语料: {path}")
            continue
        records = [json.loads(line) for line in open(path, encoding="utf-8")]
        if records and str(records[0].get("user_id", "")).startswith("#"):
            print(
                f"[decomp] 警告: {name} 的 user_id 为 #行号(导入语料), "
                f"同用户口径恒为 0, 结果仅跨用户部分有意义"
            )
        res = decompose(records)
        results[name] = res
        rr, pr = res["redundant_rows"], res["pairs"]
        print(f"\n== {name} ==  n={res['n_docs']}")
        print(
            f"重复组 {res['n_dup_groups']} 个, 成员 {res['n_dup_members']} 篇"
            f" (占 {res['dup_member_rate']:.1%})"
        )
        print(
            f"  行归属: 同用户多余行 {rr['same_user']} / 跨用户 "
            f"{rr['cross_user']}  (同用户占 "
            f"{rr['same_user_frac']:.1%})"
        )
        print(
            f"  配对归属: 同用户对 {pr['same_user']} / 跨用户对 "
            f"{pr['cross_user']}  (同用户占 "
            f"{pr['same_user_frac']:.1%})"
        )
        for d in res["top_groups"][: args.top]:
            print(
                f"    组大小 {d['size']:>3} | 用户 {d['users']} | "
                f"{d['opener']}"
            )

    if args.json:
        out = os.path.join(
            "results", "diversity", category, "dup_decomposition.json"
        )
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[decomp] 已写入 {out}")


if __name__ == "__main__":
    main()
