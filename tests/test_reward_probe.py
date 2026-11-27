"""Exercise Stage 2 cache replay with test split keys and real reward functions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.sidreasoner_reward_probe import analyze, main
from src.eval.sidreasoner_rollout_cache import CACHE_VERSION, generate_batch
from src.reward.sid_rewards.base import BEAM_SEP

GT = "<a_1><b_2><c_3>"
MISS = "<a_1><b_2><c_4>"
SHARED = "one two three four five"
RARE = "alpha one gamma delta epsilon"


def query(index, cot_beams):
    return {
        "extra_info": {"split": "test", "index": index},
        "data_source": "rec/Video_Games",
        "ground_truth": GT,
        "history": [{"sid": f"<a_{index}><b_0><c_0>", "title": f"Game {index}"}],
        "rollouts": [
            {
                "solution_str": f"{cot}</think>\n\n{sid}",
                "beam_search_results": beams,
            }
            for cot, sid, beams in cot_beams
        ],
    }


def test_probe_replays_test_split_with_optional_rollout_beam(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET", "games")
    monkeypatch.setenv("SID_SET", "official")
    embeddings = tmp_path / "users.npz"
    np.savez(
        embeddings, split=["test", "test"], index=[0, 1],
        embeddings=[[1.0, 0.0], [0.8, 0.6]], history_items=[[0], [1]],
    )
    meta = {
        "cache_version": CACHE_VERSION, "category": "Video_Games", "sid_suffix": "",
        "model": "unused", "rollout_n": 2, "sid_length": 3, "n_queries": 2,
    }
    batch = {"queries": [
        query(0, [(RARE, GT, [GT, MISS]), (SHARED, GT, ["", GT])]),
        query(1, [(SHARED, GT, [GT, MISS]), (SHARED, GT, [GT, MISS])]),
    ]}
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({"meta": meta}) + "\n" + json.dumps(batch) + "\n")
    cfg = {
        "data": {"categories": {"games": "Video_Games"}},
        "rl": {
            "reward_kwargs": {},
            "inter_query_diversity": {
                "user_embedding_path": str(embeddings),
                "ngram_n": 2, "top_k_neighbors": 1, "seed": 0,
                "lambda_init": 0.5, "sigma": 1.0,
            },
        },
    }
    args = ["--cache", str(cache), "--reward", "ngram"]
    combined = main(args + ["--correctness", "rollout-beam", "--out", str(tmp_path / "mixed.json")], cfg)
    pure = main(args + ["--out", str(tmp_path / "pure.json")], cfg)
    unigram = main(args + ["--ngram-n", "1", "--out", str(tmp_path / "unigram.json")], cfg)
    report = json.loads(combined.read_text())
    assert report["summary"]["effective_comparison_coverage"]["rate"] == 1
    assert report["summary"]["grpo_zero_variance_rate"] == 0.5
    assert report["summary"]["grpo_mean_within_group_reward_std"] == pytest.approx(0.125)
    assert report["summary"]["target_rank_concordance"] == {
        "rate": 1, "informative_queries": 1, "informative_pairs": 1,
    }
    assert report["reward_specific"]["zero_overlap"] == {
        "rate": 0.25, "zero_overlap_pairs": 1, "compared_pairs": 4,
    }
    assert report["eligible_case_count"] == 4
    rare, shared = report["cases"][:2]
    assert rare["reward"] == 3.5
    assert shared["target_rank"] == 2
    reference = rare["references"][0]
    assert reference["history"] == batch["queries"][1]["history"]
    assert reference["cot"] == SHARED
    assert reference["pair_td"] == 1.0
    assert reference["reward"] == 3.0
    assert json.loads(pure.read_text())["cases"][0]["reward"] == 0.5
    assert json.loads(unigram.read_text())["reward_specific"]["zero_overlap"]["rate"] == 0


def test_cases_are_bounded_and_keep_only_eligible_candidates():
    all_rows = [(SHARED, GT, [GT])] * 16
    batch = {"queries": [query(i, all_rows) for i in range(8)]}

    def scores(rows):
        result = []
        for i in range(len(rows)):
            reference = 16 if i < 16 else 0
            result.append({
                "score": 0.0, "cot_ok": i != 0, "success": i != 1,
                "n_refs": int(i != 2), "references": [{"row_index": reference, "td": 0.0}],
            })
        return result

    meta = {"rollout_n": 16, "n_queries": 8, "sid_length": 3}
    first = analyze([batch], meta, scores, "ngram", case_seed=19)
    assert first == analyze([batch], meta, scores, "ngram", case_seed=19)
    assert first["eligible_case_count"] == 125
    sampled = {(c["query"]["index"], c["rollout_index"]) for c in first["cases"]}
    assert len(sampled) == 100
    assert sampled.isdisjoint({(0, 0), (0, 1), (0, 2)})
    assert first["reward_specific"]["zero_overlap"]["compared_pairs"] == 125


def test_rollout_generation_keeps_query_order_and_beam_positions():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            return [9] if text == "</think>" else [8]

        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens
            return str(ids)

    class Engine:
        def generate(self, inputs, sampling, use_tqdm):
            assert [row["prompt_token_ids"] for row in inputs] == [[1]] * 16 + [[2]] * 16
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[i, 9, 8, 10])])
                    for i in range(len(inputs))]

    def truncate(ids, marker, newlines):
        assert marker == [9] and newlines == {8}
        return ids[:3], True

    def beam_search(engine, tokenizer, prompts, sampling, *, beam_width, depth):
        assert beam_width == 4 and depth == 3
        assert prompts[16] == [2, 16, 9, 8]
        return [f"{BEAM_SEP}\n{GT}"] * len(prompts)

    groups = generate_batch(
        Engine(), Tokenizer(), [[1], [2]], None, 16, 4, 3, truncate, beam_search
    )
    assert [len(group) for group in groups] == [16, 16]
    assert groups[1][0]["solution_str"] == "[16, 9, 8, 10]"
    assert groups[1][0]["beam_search_results"] == ["", GT]
