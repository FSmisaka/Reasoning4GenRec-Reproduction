"""Reward 组内区分度探针(Stage 3 RL reward 验证工具)。

目的: 抽一个 batch(B 个 query x G 条采样 rollout), 调用任意 reward 入口,
报告**每个 GRPO 组内**每条 rollout 的总 reward 与各分量(如 BLEU/多样性项),
回答"组内分数是否全相同、reward 有无区分度"。

复用方式(对后续任何新 reward 通用):
1. preset: --reward bleu(默认, 跨 Query 多样性)/ ngram(同上的 n-gram
   Jaccard 度量, 与 rl_ngram.sh 一致)/ llmjudge(LLM-as-a-Judge 语义相似度,
   与 rl_beam_llmjudge.sh 一致)/ niche(小众物品多样性, 与 rl_beam_niche.sh
   一致, 需先 make <dataset>-sidreasoner-niche-data)/ steprule(论文原版);
2. custom: --reward custom --reward-entry <入口.py> --reward-name <函数名>
   [--reward-kwargs k=v ...] [--per-row](逐行 naive API);
3. --batch-json <旧探针输出>: 复用上次生成的 batch, 只换 reward 重跑,
   不再重复生成(新 reward 对比实验的推荐用法);
4. --judge-meta N: LLM-Judge 元评测。组内/跨用户各抽样 CoT 对, 记录
   judge 的逐维打分与相似度, 写入 payload["judge_pairs"](含历史与原文),
   并在探针 JSON 旁生成 <name>.judge_pairs.md 逐对明细(含 judge 原始输出)。

输出(默认 results/reward_probe/<Category>_stage3_rl_<label>_probe.json,
命名对齐 Stage 3 实验名后缀): meta / summary(组内区分度统计) /
groups(逐组逐 rollout 的 reward 分解) / batch(原始 rollout, 供复用)。

启动(服务器, 需 vLLM):
    DATASET=games .venv/bin/python -m src.eval.sidreasoner_reward_probe
    # 或
    make games-sidreasoner-reward-probe ARGS="--n-queries 32 --rollout-n 8"
    # MODEL= 可覆盖待探测策略(默认 Stage 2 activation checkpoint = RL 起点)
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

from config import load_config
from src.train.common import resolve_category

R4GR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# reward 入口调用(batch / per-row 两种 API)
# --------------------------------------------------------------------------- #


def load_reward_fn(entry_path: str, fn_name: str):
    entry_path = os.path.abspath(entry_path)
    if not os.path.exists(entry_path):
        raise FileNotFoundError(f"reward 入口不存在: {entry_path}")
    spec = importlib.util.spec_from_file_location("reward_probe_entry", entry_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, fn_name):
        raise AttributeError(f"{entry_path} 中没有函数 {fn_name}")
    return getattr(module, fn_name)


def normalize_results(raw, n: int) -> list:
    """reward 返回值统一为 list[dict](float -> {'score': v})。"""
    out = []
    for i, item in enumerate(raw if isinstance(raw, (list, tuple)) else [raw]):
        out.append(item if isinstance(item, dict) else {"score": float(item)})
    if len(out) != n:
        raise ValueError(f"reward 返回 {len(out)} 条, 与 batch {n} 条不符。")
    return out


def call_reward(fn, rows, per_row: bool = False, kwargs: dict = None) -> list:
    """按 API 形态调用 reward 入口。"""
    kwargs = dict(kwargs or {})
    data_sources = [r["data_source"] for r in rows]
    sols = [r["solution_str"] for r in rows]
    gts = [r["ground_truth"] for r in rows]
    extras = [r["extra_info"] for r in rows]
    if per_row:
        raw = [
            fn(
                data_source=data_sources[i],
                solution_str=sols[i],
                ground_truth=gts[i],
                extra_info=extras[i],
                **kwargs,
            )
            for i in range(len(rows))
        ]
    else:
        raw = fn(
            data_sources=data_sources,
            solution_strs=sols,
            ground_truths=gts,
            extra_infos=extras,
            **kwargs,
        )
    return normalize_results(raw, len(rows))


# --------------------------------------------------------------------------- #
# 组内区分度统计(纯逻辑, 可单测)
# --------------------------------------------------------------------------- #

_DEFAULT_COMPONENTS = (
    "score", "base_reward", "td", "intrinsic",
    # niche(similarity="niche")的逐分量诊断; 其他 reward 无这些键, 自动跳过
    "niche_c1", "niche_c2", "niche_c3",
)


def component_stats(values: list) -> dict:
    arr = np.asarray([v for v in values if isinstance(v, (int, float))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    distinct = len(set(round(float(v), 9) for v in arr))
    return {
        "n": int(arr.size),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "distinct": int(distinct),
        "zero_variance": bool(distinct <= 1),
    }


def group_rows(rows: list) -> dict:
    """按 (split, index) 分组, 保持首次出现顺序。"""
    groups = {}
    for i, r in enumerate(rows):
        key = (str(r["extra_info"]["split"]), int(r["extra_info"]["index"]))
        groups.setdefault(key, []).append(i)
    return groups


def within_group_cot_similarity(cots: list, bleu) -> float:
    """组内成对 CoT 的 BLEU 相似度均值(诊断 CoT 坍缩: 越高越雷同)。"""
    sims = []
    for i in range(len(cots)):
        for j in range(i + 1, len(cots)):
            a, b = cots[i], cots[j]
            if a and b:
                sims.append(bleu.similarity(a, b))
    return float(np.mean(sims)) if sims else float("nan")


def aggregate(rows: list, results: list, components=_DEFAULT_COMPONENTS) -> dict:
    """产出逐组分解 + 全局组内区分度 summary。"""
    groups = group_rows(rows)
    group_reports = []
    comp_group_stds = defaultdict(list)
    comp_zero_var = defaultdict(lambda: [0, 0])

    for key, idxs in groups.items():
        rollouts = []
        for j, r in enumerate(idxs):
            rec = {"j": j}
            for c in components:
                if c in results[r]:
                    rec[c] = results[r][c]
            if "success" in results[r]:
                rec["success"] = int(results[r]["success"])
            if "cot_ok" in results[r]:
                rec["cot_ok"] = int(results[r]["cot_ok"])
            rec["cot_chars"] = len((rows[r]["solution_str"] or "").split("</think>")[0])
            rollouts.append(rec)

        stats = {}
        for c in components:
            vals = [rl.get(c) for rl in rollouts if c in rl]
            if vals:
                s = component_stats(vals)
                stats[c] = s
                comp_group_stds[c].append(s["std"])
                comp_zero_var[c][0] += int(s["zero_variance"])
                comp_zero_var[c][1] += 1

        from src.reward.interdiv.bleu import DiverBleu
        from src.reward.interdiv.cot_utils import extract_cot

        cots = [extract_cot(rows[r]["solution_str"]) or "" for r in idxs]
        n_refs = results[idxs[0]].get("n_refs")
        cot_ok_rate = float(
            np.mean([results[r].get("cot_ok", 0) for r in idxs])
        )
        td_all_zero = all(float(rl.get("td", 0.0)) == 0.0 for rl in rollouts)
        if not td_all_zero:
            reason = None
        elif n_refs == 0:
            reason = "no_valid_reference"       # 邻居无成功 rollout / 无可用 CoT
        elif cot_ok_rate == 0.0:
            reason = "no_usable_cot"            # 本组 CoT 全部不可解析(截断等)
        else:
            reason = "cot_identical_to_refs"    # CoT 与参考完全一致(BLEU=1 -> td=0)
        group_reports.append(
            {
                "query": {"split": key[0], "index": key[1]},
                "n_refs": n_refs,
                "cot_ok_rate": cot_ok_rate,
                "td_all_zero_reason": reason,
                "rollouts": rollouts,
                "within_group_cot_sim": within_group_cot_similarity(cots, DiverBleu()),
                "component_stats": stats,
            }
        )

    summary = {
        "n_groups": len(groups),
        "rollout_n_per_group": (
            int(np.mean([len(g["rollouts"]) for g in group_reports])) if group_reports else 0
        ),
        "per_component": {
            c: {
                "mean_group_std": float(np.mean(stds)) if stds else None,
                "frac_groups_zero_variance": (
                    float(zv[0] / zv[1]) if zv[1] else None
                ),
            }
            for c, stds in comp_group_stds.items()
            for zv in [comp_zero_var[c]]
        },
        "mean_within_group_cot_sim": float(
            np.nanmean([g["within_group_cot_sim"] for g in group_reports])
        )
        if group_reports
        else None,
        "success_rate": float(
            np.mean(
                [rl.get("success", 0) for g in group_reports for rl in g["rollouts"]]
            )
        )
        if group_reports
        else None,
        "mean_cot_ok_rate": float(
            np.mean([g["cot_ok_rate"] for g in group_reports])
        )
        if group_reports
        else None,
        "mean_n_refs": float(
            np.mean([g["n_refs"] for g in group_reports if g["n_refs"] is not None])
        )
        if any(g["n_refs"] is not None for g in group_reports)
        else None,
        "groups_td_all_zero": {
            "total": sum(1 for g in group_reports if g["td_all_zero_reason"] is not None),
            "no_valid_reference": sum(
                1 for g in group_reports
                if g["td_all_zero_reason"] == "no_valid_reference"
            ),
            "no_usable_cot": sum(
                1 for g in group_reports
                if g["td_all_zero_reason"] == "no_usable_cot"
            ),
            "cot_identical_to_refs": sum(
                1 for g in group_reports
                if g["td_all_zero_reason"] == "cot_identical_to_refs"
            ),
        },
    }
    return {"summary": summary, "groups": group_reports}


# --------------------------------------------------------------------------- #
# batch 生成(vLLM 采样) / 复用
# --------------------------------------------------------------------------- #


def default_stage2_checkpoint(cfg, category: str) -> str:
    from src.data.sid_switch import sid_suffix

    sfx = sid_suffix(cfg["data"])
    return os.path.join(
        "runs", "sidreasoner", f"{category}_stage2_activation{sfx}", "final_checkpoint"
    )


def _jsonable(obj):
    """numpy 标量/数组 -> python 原生类型(parquet extra_info 常带 np.int64)。"""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_rows_from_parquet(cfg, split: str, n_queries: int, seed: int) -> list:
    """按 seed 抽 n_queries 个 query, 展开为 prompt/ground_truth/extra_info。"""
    import pandas as pd

    rl_cfg = cfg["rl"]
    category = resolve_category(cfg["data"])
    data_dir = rl_cfg.get("data_dir") or os.path.join(
        cfg["data"]["root"], "rec_reasoning_verl", category
    )
    path = os.path.join(data_dir, f"{split}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}")
    df = pd.read_parquet(path, columns=["prompt", "extra_info", "reward_model", "data_source"])
    rng = random.Random(seed)
    picks = rng.sample(range(len(df)), min(n_queries, len(df)))

    rows = []
    for i in picks:
        rows.append(
            {
                "prompt_messages": _jsonable(list(df.iloc[i]["prompt"])),
                "data_source": str(df.iloc[i]["data_source"]),
                "ground_truth": str(df.iloc[i]["reward_model"]["ground_truth"]),
                "extra_info": _jsonable(dict(df.iloc[i]["extra_info"])),
            }
        )
    return rows


def generate_rollouts(rows, model_path, rollout_n, temperature, top_p, max_tokens, seed, gpu_mem):
    """对每个 query 采样 G 条完整 rollout(CoT + SID, 训练式自由解码)。"""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = [
        tokenizer.apply_chat_template(
            r["prompt_messages"], add_generation_prompt=True, tokenize=False
        )
        for r in rows
    ]
    llm = LLM(
        model=model_path,
        max_model_len=2048,
        max_num_seqs=32,
        dtype="bfloat16",
        gpu_memory_utilization=float(gpu_mem),
        tensor_parallel_size=1,
    )
    sp = SamplingParams(
        n=int(rollout_n),
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        seed=int(seed),
    )
    outs = llm.generate(prompts, sp)
    rollouts = []  # group-major: 第 i 个 query 的 G 条
    for out in outs:
        texts = [o.text for o in out.outputs]
        if len(texts) != int(rollout_n):
            raise RuntimeError(f"vLLM 返回 {len(texts)} 条, 期望 {rollout_n}。")
        rollouts.append(texts)
    return rollouts


def expand_rows(rows, rollouts) -> list:
    flat = []
    for r, texts in zip(rows, rollouts):
        for text in texts:
            flat.append(
                {
                    "data_source": r["data_source"],
                    "solution_str": text,
                    "ground_truth": r["ground_truth"],
                    "extra_info": r["extra_info"],
                }
            )
    return flat


# --------------------------------------------------------------------------- #
# preset
# --------------------------------------------------------------------------- #


def preset_kwargs(cfg, preset: str, out_dir: str) -> tuple:
    """返回 (entry_path, fn_name, kwargs, per_row)。"""
    iqd = cfg["rl"].get("inter_query_diversity", {}) or {}
    category = resolve_category(cfg["data"])
    if preset in ("bleu", "ngram", "llmjudge", "niche"):
        enc = iqd.get("user_encoder_type", "sasrec")
        emb = iqd.get("user_embedding_path") or os.path.join(
            "runs", "sidreasoner", "user_embed", f"{category}_{enc}.npz"
        )
        niche_dir = os.path.join(cfg["data"]["root"], "niche", category)
        kwargs = {
            "correctness_reward_type": "steprule",
            "combine": iqd.get("combine", "add"),
            "enable_inter_query_diversity": True,
            "user_embedding_path": emb,
            "top_k_neighbors": iqd.get("top_k_neighbors", 8),
            "lambda_init": iqd.get("lambda_init", 0.2),
            "lambda_final": iqd.get("lambda_final", 0.0),
            "decay_steps": iqd.get("decay_steps", 300),
            "gamma": iqd.get("gamma", 1.0),
            "sigma": iqd.get("sigma", 0.5),
            "seed": iqd.get("seed", 0),
            "success_mode": iqd.get("success_mode", "full"),
            "reference_mode": iqd.get("reference_mode", "full"),
            "bleu_smoothing": iqd.get("bleu_smoothing", "none"),
            "td_norm": iqd.get("td_norm", "none"),
            # ngram/llmjudge/niche preset 强制对应度量(与 rl_ngram.sh /
            # rl_beam_llmjudge.sh / rl_beam_niche.sh 一致); bleu preset 按
            # config(与 rl-bleu 一致, 便于跟随训练配置)
            "similarity": (
                {
                    "ngram": "ngram_jaccard",
                    "llmjudge": "llmjudge",
                    "niche": "niche",
                }.get(preset)
                or iqd.get("similarity", "bleu")
            ),
            "ngram_n": iqd.get("ngram_n", 1),
            "judge_base_url": iqd.get("judge_base_url", "http://localhost:8000/v1"),
            "judge_model": iqd.get("judge_model", "Qwen3-1.7B"),
            "judge_concurrency": iqd.get("judge_concurrency", 32),
            # niche(小众物品)多样性: 离线产物路径与方法(其他度量忽略)
            "niche_dict_path": iqd.get("niche_dict_path")
            or os.path.join(niche_dir, "popularity.json"),
            "niche_lexstats_path": iqd.get("niche_lexstats_path")
            or os.path.join(niche_dir, "cot_lexstats.json"),
            "niche_sif_path": iqd.get("niche_sif_path")
            or os.path.join(niche_dir, "sif_words.npz"),
            "niche_summary_method": iqd.get("niche_summary_method", "sif"),
            "niche_embed_model": iqd.get("niche_embed_model"),
            "niche_embed_device": iqd.get("niche_embed_device", "cpu"),
            "niche_embed_batch_size": iqd.get("niche_embed_batch_size", 64),
            "niche_embed_num_threads": iqd.get("niche_embed_num_threads", 8),
            "niche_only_success": iqd.get("niche_only_success", True),
            # 探针的 interdiv 日志独立落盘, 不污染训练的 interdiv_metrics.jsonl
            "metrics_path": os.path.join(out_dir, "interdiv_probe_metrics.jsonl"),
        }
        return (
            os.path.join(R4GR_ROOT, "src", "reward", "direct_recommendation_interdiv.py"),
            "rule_base_reward",
            kwargs,
            False,
        )
    if preset == "steprule":
        return (
            os.path.join(R4GR_ROOT, "src", "reward", "direct_recommendation_steprule.py"),
            "rule_base_reward",
            {},
            True,
        )
    raise ValueError(
        f"未知 preset '{preset}'(bleu | ngram | niche | steprule; 自定义用 --reward custom)"
    )


def parse_kv(pairs: list) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--reward-kwargs 需要 k=v 形式, 收到 {p!r}")
        k, _, v = p.partition("=")
        if not k:
            raise ValueError(f"--reward-kwargs 键为空: {p!r}")
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        if isinstance(v, str) and v.lower() in ("true", "false"):
            v = v.lower() == "true"
        out[k] = v
    return out


# --------------------------------------------------------------------------- #
# LLM-Judge 元评测: 抽样 CoT 对, 记录 judge 的逐维打分
# --------------------------------------------------------------------------- #


def judge_meta_eval(rows: list, n_pairs: int, judge, history, seed: int = 0) -> dict:
    """组内 / 跨用户各半抽样 CoT 对, 返回逐对逐维打分与汇总。"""
    from src.reward.interdiv.cot_utils import extract_cot
    from src.reward.interdiv.llmjudge import render_block

    cots = {i: extract_cot(r["solution_str"]) for i, r in enumerate(rows)}
    usable = {
        k: [i for i in idxs if cots[i] and len(cots[i].split()) >= 5]
        for k, idxs in group_rows(rows).items()
    }
    usable = {k: v for k, v in usable.items() if v}
    keys = list(usable)
    rng = random.Random(seed)

    pairs = []
    for k in keys:  # 组内(同 query 不同 rollout)
        if len(usable[k]) >= 2:
            i, j = rng.sample(usable[k], 2)
            pairs.append(("within_user", k, k, i, j))
    for _ in range(len(pairs)):  # 跨用户
        if len(keys) < 2:
            break
        ka, kb = rng.sample(keys, 2)
        pairs.append(
            ("cross_user", ka, kb, rng.choice(usable[ka]), rng.choice(usable[kb]))
        )
    rng.shuffle(pairs)
    pairs = pairs[: int(n_pairs)]
    if not pairs:
        return {"pairs": [], "summary": None}

    def block(key, i):
        return render_block(history.lines(*key), cots[i])

    scores = judge.judge_pairs(
        [(block(ka, i), block(kb, j)) for _, ka, kb, i, j in pairs]
    )
    raws = getattr(judge, "last_raw", [""] * len(pairs))
    parsed = getattr(judge, "last_parsed", [True] * len(pairs))
    records = []
    for (kind, ka, kb, i, j), s, raw, ok in zip(pairs, scores, raws, parsed):
        sim = sum(s.values()) / (4.0 * len(s))
        records.append(
            {
                "type": kind,
                "query_a": {"split": ka[0], "index": ka[1]},
                "query_b": {"split": kb[0], "index": kb[1]},
                "history_a": history.lines(*ka),
                "history_b": history.lines(*kb),
                "cot_a": cots[i],
                "cot_b": cots[j],
                "scores": s,
                "similarity": sim,
                "diversity": 1.0 - sim,
                "parsed": bool(ok),
                "raw": raw,
            }
        )
    summary = {}
    for kind in ("within_user", "cross_user"):
        ss = [r for r in records if r["type"] == kind]
        if ss:
            summary[kind] = {
                "n": len(ss),
                "mean_similarity": float(np.mean([r["similarity"] for r in ss])),
                **{
                    f"mean_{k}": float(np.mean([r["scores"][k] for r in ss]))
                    for k in ss[0]["scores"]
                },
            }
    return {"pairs": records, "summary": summary}


def _fmt_query(q) -> str:
    return f"{q['split']}[{q['index']}]"


def judge_report_md(judge_pairs: dict, info: dict) -> str:
    """逐对明细的 Markdown 报告(历史 + CoT + 三维打分 + judge 原始输出)。"""
    recs = judge_pairs["pairs"]
    out = [
        "# LLM-Judge 元评测",
        "",
        f"- 数据集: {info['category']}",
        f"- judge: {info['model']} @ {info['base_url']}",
        f"- 对数: {len(recs)}",
        f"- judge 调用: {info['calls']}，解析失败回退: {info['fallback']}",
        "",
    ]
    if judge_pairs["summary"]:
        out += [
            "## 分组均值",
            "",
            "| type | n | evidence | strategy | profile | sim | td |",
            "|---|---|---|---|---|---|---|",
        ]
        for kind, st in judge_pairs["summary"].items():
            out.append(
                f"| {kind} | {st['n']} | {st['mean_evidence']:.2f} | "
                f"{st['mean_strategy']:.2f} | {st['mean_profile']:.2f} | "
                f"{st['mean_similarity']:.3f} | {1 - st['mean_similarity']:.3f} |"
            )
        out.append("")
    out += [
        "## 逐对打分",
        "",
        "| # | type | query A | query B | evidence | strategy | profile | sim | td | 解析 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(recs):
        s = r["scores"]
        out.append(
            f"| {i} | {r['type']} | {_fmt_query(r['query_a'])} | "
            f"{_fmt_query(r['query_b'])} | {s['evidence']} | {s['strategy']} | "
            f"{s['profile']} | {r['similarity']:.3f} | {r['diversity']:.3f} | "
            f"{'OK' if r.get('parsed', True) else 'FALLBACK'} |"
        )
    out += ["", "## 明细", ""]
    for i, r in enumerate(recs):
        s = r["scores"]
        flag = "" if r.get("parsed", True) else "  **解析失败, 全 2 回退**"
        out += [
            f"### {i}. {r['type']}: {_fmt_query(r['query_a'])} vs {_fmt_query(r['query_b'])}",
            "",
            f"evidence **{s['evidence']}** / strategy **{s['strategy']}** / "
            f"profile **{s['profile']}** → sim {r['similarity']:.3f} / "
            f"td {r['diversity']:.3f}{flag}",
            "",
            "**User A history**",
            "````",
            *r["history_a"],
            "````",
            "",
            "**CoT A**",
            "````",
            r["cot_a"],
            "````",
            "",
            "**User B history**",
            "````",
            *r["history_b"],
            "````",
            "",
            "**CoT B**",
            "````",
            r["cot_b"],
            "````",
            "",
            "**Judge raw output**",
            "````json",
            r.get("raw", ""),
            "````",
            "",
        ]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main(argv=None, cfg=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reward", default="bleu",
        help="preset: bleu | ngram | llmjudge | niche | steprule | custom",
    )
    parser.add_argument("--reward-entry", default=None, help="custom: reward 入口 .py 路径")
    parser.add_argument("--reward-name", default="rule_base_reward", help="custom: 入口函数名")
    parser.add_argument(
        "--reward-kwargs", action="append", default=None, help="custom: k=v, 可重复"
    )
    parser.add_argument("--per-row", action="store_true", help="custom: 逐行(naive)API")
    parser.add_argument("--label", default=None, help="输出文件名中的 reward 标签")
    parser.add_argument("--batch-json", default=None, help="复用旧探针的 batch, 跳过生成")
    parser.add_argument("--model", default=None, help="策略 checkpoint(默认 Stage 2 / MODEL=)")
    parser.add_argument("--split", default="train", help="parquet split(默认 train)")
    parser.add_argument("--n-queries", type=int, default=32, help="batch 内 query 数 B")
    parser.add_argument("--rollout-n", type=int, default=16, help="每 query 采样条数 G")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--judge-meta", type=int, default=0,
        help="LLM-Judge 元评测: 抽样 N 个 CoT 对, 记录逐维打分(0=关闭)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    cfg = cfg or load_config("sidreasoner")
    category = resolve_category(cfg["data"])
    label = args.label or (args.reward if args.reward != "custom" else "custom")
    out_path = args.out or os.path.join(
        "results", "reward_probe", f"{category}_stage3_rl_{label}_probe.json"
    )
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    # ---- batch: 生成或复用 ----
    if args.batch_json:
        prev = json.load(open(args.batch_json))
        rows = prev["batch"]["rows"]
        meta_gen = dict(prev["meta"]["generation"])
        print(f"[probe] 复用 batch: {args.batch_json}({len(rows)} 条 rollout)")
    else:
        model = args.model or os.environ.get("MODEL") or default_stage2_checkpoint(cfg, category)
        if not os.path.exists(model):
            raise FileNotFoundError(
                f"未找到策略 checkpoint: {model}(用 --model / MODEL= 指定, "
                "默认 Stage 2 activation final_checkpoint)"
            )
        picked = build_rows_from_parquet(cfg, args.split, args.n_queries, args.seed)
        print(f"[probe] {len(picked)} queries x G={args.rollout_n} 采样 @ {model}")
        rollouts = generate_rollouts(
            picked, model, args.rollout_n, args.temperature, args.top_p,
            args.max_tokens, args.seed, args.gpu_mem,
        )
        rows = expand_rows(picked, rollouts)
        meta_gen = {
            "model": model,
            "split": args.split,
            "n_queries": len(picked),
            "rollout_n": args.rollout_n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        }

    # ---- reward ----
    if args.reward == "custom":
        if not args.reward_entry:
            raise SystemExit("--reward custom 需要 --reward-entry <入口.py>")
        entry, name, kwargs, per_row = (
            args.reward_entry, args.reward_name, parse_kv(args.reward_kwargs), args.per_row
        )
    else:
        entry, name, kwargs, per_row = preset_kwargs(cfg, args.reward, out_dir)
        if args.reward_kwargs:
            kwargs.update(parse_kv(args.reward_kwargs))

    fn = load_reward_fn(entry, name)
    print(f"[probe] reward: {os.path.basename(entry)}::{name} per_row={per_row}")
    results = call_reward(fn, rows, per_row=per_row, kwargs=kwargs)

    # ---- LLM-Judge 元评测(可选) ----
    judge_pairs = {"pairs": [], "summary": None}
    if args.judge_meta > 0:
        from src.reward.interdiv.llmjudge import HistoryLookup, LLMJudge

        iqd = cfg["rl"].get("inter_query_diversity", {}) or {}
        judge = LLMJudge(
            base_url=iqd.get("judge_base_url", "http://localhost:8000/v1"),
            model=iqd.get("judge_model", "Qwen3-1.7B"),
            concurrency=iqd.get("judge_concurrency", 32),
        )
        history = HistoryLookup(cfg["data"]["root"], category)
        judge_pairs = judge_meta_eval(rows, args.judge_meta, judge, history, args.seed)
        for kind, st in (judge_pairs["summary"] or {}).items():
            print(
                f"[probe] judge {kind}: n={st['n']} sim={st['mean_similarity']:.3f} "
                f"evidence={st['mean_evidence']:.2f} "
                f"strategy={st['mean_strategy']:.2f} "
                f"profile={st['mean_profile']:.2f}"
            )
        print(f"[probe] judge calls={judge.n_calls} fallback={judge.n_fallback}")
        print("[probe]  #  type         query A        query B        E S P   sim   解析")
        for i, r in enumerate(judge_pairs["pairs"]):
            s = r["scores"]
            print(
                f"[probe] {i:2d}  {r['type']:11s} {_fmt_query(r['query_a']):14s} "
                f"{_fmt_query(r['query_b']):14s} {s['evidence']} {s['strategy']} "
                f"{s['profile']}  {r['similarity']:.3f}  "
                f"{'OK' if r.get('parsed', True) else 'FALLBACK'}"
            )
        report_path = (
            out_path[:-5] + ".judge_pairs.md"
            if out_path.endswith(".json")
            else out_path + ".judge_pairs.md"
        )
        with open(report_path, "w") as f:
            f.write(
                judge_report_md(
                    judge_pairs,
                    {
                        "category": category,
                        "model": judge.model,
                        "base_url": judge.base_url,
                        "calls": judge.n_calls,
                        "fallback": judge.n_fallback,
                    },
                )
            )
        print(f"[probe] judge 逐对明细 -> {report_path}")

    # ---- 汇总 & 落盘 ----
    report = aggregate(rows, results)
    payload = {
        "meta": {
            "category": category,
            "reward": {
                "label": label,
                "entry": os.path.abspath(entry),
                "name": name,
                "kwargs": kwargs,
                "per_row": per_row,
            },
            "generation": meta_gen,
        },
        "summary": report["summary"],
        "groups": report["groups"],
        "judge_pairs": judge_pairs,
        "batch": {"rows": _jsonable(rows)},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    s = report["summary"]
    print(f"[probe] -> {out_path}")
    print(f"[probe] groups={s['n_groups']} G={s['rollout_n_per_group']} "
          f"success={s.get('success_rate')} cot_ok={s.get('mean_cot_ok_rate')} "
          f"n_refs={s.get('mean_n_refs')}")
    for c, st in s["per_component"].items():
        print(
            f"[probe] {c:12s} 组内std均值={st['mean_group_std']:.6g} "
            f"零方差组占比={st['frac_groups_zero_variance']:.1%}"
        )
    z = s["groups_td_all_zero"]
    if z["total"]:
        print(
            f"[probe] td 全零组: {z['total']}/{s['n_groups']} "
            f"(无有效 Reference {z['no_valid_reference']} / CoT 不可解析 "
            f"{z['no_usable_cot']} / CoT 与参考雷同 {z['cot_identical_to_refs']})"
        )
    print(f"[probe] 组内 CoT 成对 BLEU 相似度均值 = {s['mean_within_group_cot_sim']:.4f} "
          "(越高说明同组 CoT 越雷同, BLEU 项越难区分)")
    return payload


if __name__ == "__main__":
    main()
