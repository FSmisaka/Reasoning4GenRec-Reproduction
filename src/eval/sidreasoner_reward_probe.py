"""Score a Stage 2 rollout cache and report group statistics and real CoT comparisons."""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import random
from statistics import fmean, pstdev

import numpy as np

from config import load_config
from src.data.sid_switch import sid_suffix
from src.data.sidreasoner_user_embed import default_embedding_path
from src.eval.sidreasoner_rollout_cache import CACHE_VERSION, atomic_output, default_cache_path
from src.reward.direct_recommendation_interdiv import _build_base_computer, build_diversity_addon
from src.reward.interdiv.cot_utils import extract_cot, is_recommendation_success, is_usable_cot
from src.reward.interdiv.shaping import normalize_td, resolve_td_norm
from src.reward.sid_rewards.base import extract_sid_tokens
from src.train.common import resolve_category

DIVERSITY = {"ngram": "ngram_jaccard", "bleu": "bleu", "llmjudge": "llmjudge", "niche": "niche"}
CORRECTNESS = {
    "steprule": "steprule",
    "rollout-beam": "onerec_rollout_beam",
    "rollout-beam-div": "onereason_div_rollout_beam",
}
CASE_LIMIT = 100


def parse_overrides(values):
    overrides = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not key or not sep:
            raise ValueError(f"--reward-kwargs 需要 k=v: {value!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def load_custom_reward(path, name, per_row, kwargs):
    spec = importlib.util.spec_from_file_location("_offline_reward_probe", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载 reward 入口: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, name)

    def score(rows):
        inputs = [
            {
                "data_source": row["data_source"],
                "solution_str": row["solution_str"],
                "ground_truth": row["ground_truth"],
                "extra_info": {**row["extra_info"], "history": row["history"]},
            }
            for row in rows
        ]
        raw = (
            [fn(**row, **kwargs) for row in inputs]
            if per_row else fn(
                data_sources=[row["data_source"] for row in inputs],
                solution_strs=[row["solution_str"] for row in inputs],
                ground_truths=[row["ground_truth"] for row in inputs],
                extra_infos=[row["extra_info"] for row in inputs],
                **kwargs,
            )
        )
        return [item if isinstance(item, dict) else {"score": item} for item in raw]

    return score


def normalize_custom_diversity(rows, results, mode, success_mode, sid_length):
    groups = defaultdict(list)
    prepared = []
    for i, (row, result) in enumerate(zip(rows, results, strict=True)):
        extra = row["extra_info"]
        groups[(extra["split"], extra["index"])].append(i)
        result = dict(result)
        if "cot_ok" not in result:
            result["cot_ok"] = is_usable_cot(extract_cot(row["solution_str"]))
        if "success" not in result:
            result["success"] = is_recommendation_success(
                row["solution_str"], row["ground_truth"], num_sid=sid_length, mode=success_mode,
            )
        prepared.append(result)

    values = normalize_td(
        [result["score"] for result in prepared], groups.values(),
        [result["success"] and result["cot_ok"] for result in prepared], mode,
    )
    for result, value in zip(prepared, values, strict=True):
        result["score"] = float(value)
    return prepared


def configure_reward(cfg, category, meta, args):
    overrides = parse_overrides(args.reward_kwargs)
    if overrides.get("combine", "add") != "add":
        raise ValueError("离线探针的正确性与多样性 reward 使用加法合并（combine=add）。")

    rl = cfg["rl"]
    diversity_cfg = rl.get("inter_query_diversity", {})
    norm_source = overrides if {"norm", "td_norm"} & overrides.keys() else diversity_cfg
    norm = resolve_td_norm(norm_source.get("norm", norm_source.get("td_norm", "none")))
    base_kwargs = {
        **rl.get("reward_kwargs", {}), **overrides,
        "correctness_reward_type": CORRECTNESS[args.correctness],
        "num_sid_levels": meta["sid_length"],
    }
    computer, _ = _build_base_computer(base_kwargs)
    settings = {"correctness": base_kwargs}

    addon = None
    custom_score = None
    if args.diversity == "custom":
        custom_kwargs = {k: value for k, value in overrides.items() if k not in {"norm", "td_norm"}}
        custom_score = load_custom_reward(
            args.reward_entry, args.reward_name, args.per_row, custom_kwargs
        )
        success_mode = overrides.get("success_mode", diversity_cfg.get("success_mode", "full"))
        settings["diversity"] = {**custom_kwargs, "td_norm": norm, "success_mode": success_mode}
    else:
        kwargs = dict(diversity_cfg)
        kwargs["user_embedding_path"] = (
            kwargs.get("user_embedding_path")
            or default_embedding_path(category, kwargs.get("user_encoder_type", "sasrec"))
        )
        kwargs["model_path"] = meta["model"]
        kwargs.update(overrides)
        kwargs.pop("norm", None)
        kwargs["td_norm"] = norm
        kwargs["similarity"] = DIVERSITY[args.diversity]
        kwargs["combine"] = "add"
        if args.diversity == "ngram":
            kwargs["ngram_n"] = args.ngram_n or int(kwargs.get("ngram_n", 1))
        addon = build_diversity_addon(kwargs)
        settings["diversity"] = kwargs

    def score(rows):
        base_rewards = [
            computer.compute(
                data_source=row["data_source"], solution_str=row["solution_str"],
                ground_truth=row["ground_truth"], extra_info=row["extra_info"],
            )
            for row in rows
        ]
        if custom_score is not None:
            results = custom_score(rows)
            if norm != "none":
                results = normalize_custom_diversity(
                    rows, results, norm, success_mode, meta["sid_length"],
                )
            combined = []
            for result, base in zip(results, base_rewards, strict=True):
                intrinsic = float(result["score"])
                combined.append({
                    **result, "base_reward": base, "intrinsic": intrinsic, "score": base + intrinsic,
                })
            return combined
        return addon.compute_batch(rows, base_rewards, trainer_step=args.shaping_step, probe=True)

    return score, settings


def flatten_batch(batch, rollout_n):
    rows, groups = [], []
    for query in batch["queries"]:
        if len(query["rollouts"]) != rollout_n:
            raise ValueError(f"缓存 query {query['extra_info']} 的 rollout 数与元数据不符。")
        start = len(rows)
        for j, rollout in enumerate(query["rollouts"]):
            rows.append({
                "data_source": query["data_source"],
                "solution_str": rollout["solution_str"],
                "ground_truth": query["ground_truth"],
                "history": query["history"],
                "rollout_index": j,
                "extra_info": {
                    **query["extra_info"], "beam_search_results": rollout["beam_search_results"],
                },
            })
        groups.append(range(start, len(rows)))
    return rows, groups


def target_rank(row, sid_length):
    target = extract_sid_tokens(row["ground_truth"])
    beams = row["extra_info"]["beam_search_results"]
    if len(target) != sid_length:
        raise ValueError(f"缓存里的目标 SID 无效: {row['ground_truth']!r}")
    for rank, beam in enumerate(beams, 1):
        if extract_sid_tokens(beam)[:sid_length] == target:
            return rank
    return math.inf


def eligible(result):
    return result.get("cot_ok") and result.get("success") and result.get("n_refs", 0) >= 1


def describe_rollout(row, result, rank):
    return {
        "query": {key: row["extra_info"][key] for key in ("split", "index")},
        "rollout_index": row["rollout_index"],
        "history": row["history"],
        "cot": extract_cot(row["solution_str"]),
        "cot_ok": bool(result.get("cot_ok")),
        "success": bool(result.get("success")),
        "n_refs": result.get("n_refs"),
        "td": result.get("td"),
        "td_used": result.get("td_used"),
        "intrinsic": result.get("intrinsic"),
        "base_reward": result.get("base_reward"),
        "reward": float(result["score"]),
        "target_rank": rank if math.isfinite(rank) else None,
    }


def describe_case(i, rows, results, ranks):
    case = describe_rollout(rows[i], results[i], ranks[i])
    case["references"] = []
    for detail in results[i].get("references", []):
        j = detail["row_index"]
        reference = describe_rollout(rows[j], results[j], ranks[j])
        reference["user_similarity"] = detail.get("user_similarity")
        reference["pair_td"] = detail.get("td")
        case["references"].append(reference)
    return case


@dataclass
class RewardStats:
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    covered_queries: int = 0
    concordance: list[float] = field(default_factory=list)
    informative_pairs: int = 0

    def add_group(self, scores, group, candidates, control, ranks, *, has_coverage=True):
        values = [scores[i] for i in group]
        self.means.append(fmean(values))
        self.stds.append(pstdev(values))

        covered = False
        comparisons = []
        for i, j in combinations(candidates, 2):
            if control is not None and control[i] != control[j]:
                continue
            covered = True
            if ranks[i] != ranks[j]:
                comparisons.append(
                    0.5 if scores[i] == scores[j]
                    else float((scores[i] > scores[j]) == (ranks[i] < ranks[j]))
                )
        self.covered_queries += has_coverage and covered
        self.informative_pairs += len(comparisons)
        if comparisons:
            self.concordance.append(fmean(comparisons))

    def summary(self, coverage_available=True):
        n_queries = len(self.means)
        return {
            "within_group_mean": fmean(self.means),
            "within_group_std": fmean(self.stds),
            "zero_variance_rate": self.stds.count(0.0) / n_queries,
            "effective_comparison_coverage": {
                "rate": self.covered_queries / n_queries if coverage_available else None,
                "covered_queries": self.covered_queries,
            },
            "target_rank_concordance": {
                "rate": fmean(self.concordance) if self.concordance else None,
                "informative_queries": len(self.concordance),
                "informative_pairs": self.informative_pairs,
            },
        }


def analyze(batches, meta, score, reward, case_seed):
    counts = defaultdict(int)
    fields = {"diversity": "intrinsic", "correctness": "base_reward", "total": "score"}
    stats = {component: RewardStats() for component in fields}
    eligible_means, eligible_stds, cases = [], [], []
    rng = random.Random(case_seed)

    for batch in batches:
        rows, groups = flatten_batch(batch, meta["rollout_n"])
        results = score(rows)
        if len(results) != len(rows):
            raise ValueError("reward 返回的分数数量与 rollout 数量不一致。")
        # The training reward manager stores scores as float32 before GRPO.
        reward_values = {}
        for component, key in fields.items():
            values = np.asarray([result[key] for result in results], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{component} reward 必须是有限数值。")
            reward_values[component] = values.tolist()
        ranks = [target_rank(row, meta["sid_length"]) for row in rows]

        for group in groups:
            counts["queries"] += 1
            has_reference_diagnostics = any("n_refs" in results[i] for i in group)
            counts["queries_with_diagnostics"] += has_reference_diagnostics
            valid = [i for i in group if eligible(results[i])]
            counts["queries_with_eligible_reference"] += bool(valid)
            diversity = reward_values["diversity"]
            correctness = reward_values["correctness"]
            stats["diversity"].add_group(
                diversity, group, valid if has_reference_diagnostics else group,
                correctness, ranks, has_coverage=has_reference_diagnostics,
            )
            stats["correctness"].add_group(correctness, group, group, diversity, ranks)
            stats["total"].add_group(reward_values["total"], group, group, None, ranks)

            if reward == "ngram" and valid:
                eligible_scores = [diversity[i] for i in valid]
                eligible_means.append(fmean(eligible_scores))
                eligible_stds.append(pstdev(eligible_scores))

            for i in valid:
                details = results[i].get("references", [])
                if reward == "ngram":
                    counts["ngram_pairs"] += len(details)
                    counts["zero_overlap_pairs"] += sum(detail["td"] == 1.0 for detail in details)
                if not details:
                    continue
                counts["eligible_cases"] += 1
                # Reservoir sampling gives every eligible comparison the same chance of inclusion.
                if len(cases) < CASE_LIMIT:
                    cases.append(describe_case(i, rows, results, ranks))
                else:
                    slot = rng.randrange(counts["eligible_cases"])
                    if slot < CASE_LIMIT:
                        cases[slot] = describe_case(i, rows, results, ranks)

    if counts["queries"] != meta["n_queries"] or not counts["queries"]:
        raise ValueError("缓存里的 query 数与元数据不符，或缓存为空。")

    summary = {
        "n_queries": counts["queries"],
        "n_rollouts": counts["queries"] * meta["rollout_n"],
        "diversity": stats["diversity"].summary(bool(counts["queries_with_diagnostics"])),
        "correctness": stats["correctness"].summary(),
        "total": stats["total"].summary(),
    }
    summary["diversity"]["effective_comparison_coverage"].update({
        "queries_with_eligible_reference": counts["queries_with_eligible_reference"],
        "queries_with_diagnostics": counts["queries_with_diagnostics"],
    })
    specific = {}
    if reward == "ngram":
        total = counts["ngram_pairs"]
        specific["zero_overlap"] = {
            "rate": counts["zero_overlap_pairs"] / total if total else None,
            "zero_overlap_pairs": counts["zero_overlap_pairs"],
            "compared_pairs": total,
        }
        specific.update({
            "eligible_within_group_mean": fmean(eligible_means) if eligible_means else None,
            "eligible_within_group_std": fmean(eligible_stds) if eligible_stds else None,
            "eligible_queries": len(eligible_means),
        })
    return {
        "summary": summary, "reward_specific": specific,
        "eligible_case_count": counts["eligible_cases"], "cases": cases,
    }


def main(argv=None, cfg=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--diversity", choices=[*DIVERSITY, "custom"], required=True, help="多样性奖励。",
    )
    parser.add_argument(
        "--correctness", choices=CORRECTNESS, required=True, help="正确性奖励。",
    )
    parser.add_argument("--ngram-n", type=int)
    parser.add_argument("--shaping-step", type=int, default=0)
    parser.add_argument("--reward-entry", type=Path)
    parser.add_argument("--reward-name", default="rule_base_reward")
    parser.add_argument("--reward-kwargs", action="append", default=[])
    parser.add_argument("--per-row", action="store_true")
    parser.add_argument("--case-seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.diversity == "custom" and not args.reward_entry:
        parser.error("--diversity custom 需要 --reward-entry。")
    if args.ngram_n is not None and (args.diversity != "ngram" or args.ngram_n < 1):
        parser.error("--ngram-n 仅用于 ngram，且必须是正整数。")
    if args.shaping_step < 0:
        parser.error("--shaping-step 必须非负。")

    cfg = cfg or load_config("sidreasoner")
    category = resolve_category(cfg["data"])
    suffix = sid_suffix(cfg["data"])
    cache = args.cache or default_cache_path(cfg, category)
    reward = args.diversity
    with cache.open(encoding="utf-8") as handle:
        meta = json.loads(handle.readline())["meta"]
        if (
            meta["cache_version"] != CACHE_VERSION
            or meta["category"] != category
            or meta["sid_suffix"] != suffix
        ):
            raise ValueError("缓存的版本、数据集或 SID 设置与当前探针不匹配。")
        score, settings = configure_reward(cfg, category, meta, args)
        report = analyze((json.loads(line) for line in handle), meta, score, reward, args.case_seed)

    label = reward
    if args.diversity == "ngram":
        label += f"_n{settings['diversity']['ngram_n']}"
    label = f"{args.correctness}_{label}"
    out = args.out or Path("results/reward_probe") / f"{category}{suffix}_stage2_test_{label}.json"
    if cache.resolve() == out.resolve():
        parser.error("--out 需要使用与 rollout 缓存不同的路径。")
    with atomic_output(out) as handle:
        json.dump({
            "meta": {
                "cache": str(cache), "generation": meta, "reward": reward,
                "diversity": args.diversity,
                "correctness": args.correctness, "settings": settings,
                "shaping_step": args.shaping_step, "case_seed": args.case_seed,
                **({"custom": {
                    "entry": str(args.reward_entry), "name": args.reward_name,
                    "per_row": args.per_row,
                }} if args.diversity == "custom" else {}),
            },
            **report,
        }, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"[reward-probe] {out}")
    return out


if __name__ == "__main__":
    main()
