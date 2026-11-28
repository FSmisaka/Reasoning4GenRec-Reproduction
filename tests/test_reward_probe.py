"""Exercise Stage 2 cache replay with test split keys and real reward functions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.eval.sidreasoner_reward_probe import analyze, configure_reward, main
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


def test_probe_replays_test_split_with_rollout_beam(tmp_path, monkeypatch):
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
                "lambda_init": 0.5, "sigma": 1.0, "combine": "mul",
            },
        },
    }
    args = ["--cache", str(cache), "--diversity", "ngram", "--correctness", "rollout-beam"]
    combined = main(args + ["--out", str(tmp_path / "mixed.json")], cfg)
    unigram = main(args + ["--ngram-n", "1", "--out", str(tmp_path / "unigram.json")], cfg)
    report = json.loads(combined.read_text())
    assert report["meta"]["diversity"] == "ngram"
    assert report["meta"]["correctness"] == "rollout-beam"
    summary = report["summary"]
    for component, (mean, std, zero_rate, coverage) in {
        "diversity": (0.125, 0.125, 0.5, 1.0),
        "correctness": (3.0, 0.0, 1.0, 0.5),
        "total": (3.125, 0.125, 0.5, 1.0),
    }.items():
        assert summary[component]["within_group_mean"] == pytest.approx(mean)
        assert summary[component]["within_group_std"] == pytest.approx(std)
        assert summary[component]["zero_variance_rate"] == zero_rate
        assert summary[component]["effective_comparison_coverage"]["rate"] == coverage
    rank_stats = {
        "rate": 1, "informative_queries": 1, "informative_pairs": 1,
    }
    assert summary["diversity"]["target_rank_concordance"] == rank_stats
    assert summary["total"]["target_rank_concordance"] == rank_stats
    assert summary["correctness"]["target_rank_concordance"]["rate"] is None
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
    assert json.loads(unigram.read_text())["reward_specific"]["zero_overlap"]["rate"] == 0


@pytest.mark.parametrize(("args", "missing"), [
    (["--diversity", "ngram"], "--correctness"),
    (["--correctness", "rollout-beam"], "--diversity"),
])
def test_probe_requires_both_reward_components(args, missing, capsys):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert f"the following arguments are required: {missing}" in capsys.readouterr().err


@pytest.mark.parametrize(("diversity", "correctness", "invalid_arg", "choices"), [
    ("None", "rollout-beam", "--diversity", ("ngram", "bleu", "llmjudge", "niche", "custom")),
    ("ngram", "None", "--correctness", ("steprule", "rollout-beam", "rollout-beam-div")),
    ("typo", "steprule", "--diversity", ("ngram", "bleu", "llmjudge", "niche", "custom")),
])
def test_probe_lists_legal_values(diversity, correctness, invalid_arg, choices, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--diversity", diversity, "--correctness", correctness])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert f"argument {invalid_arg}: invalid choice" in message
    assert f"(choose from {', '.join(choices)})" in message.replace("'", "")


def test_custom_diversity_is_added_to_selected_correctness(tmp_path):
    entry = tmp_path / "custom.py"
    entry.write_text(
        "def score(solution_strs, **kwargs):\n"
        "    return [{'score': 0.5, 'n_refs': 1} for _ in solution_strs]\n"
    )
    args = SimpleNamespace(
        diversity="custom", correctness="rollout-beam", reward_entry=entry,
        reward_name="score", per_row=False, reward_kwargs=[],
    )
    score, _ = configure_reward({"rl": {}}, "Video_Games", {"sid_length": 3}, args)
    result = score([{
        "data_source": "rec/Video_Games",
        "solution_str": f"{SHARED}</think>\n\n{GT}",
        "ground_truth": GT,
        "extra_info": {"beam_search_results": [GT]},
        "history": [],
    }])
    assert result == [{"base_reward": 3.0, "intrinsic": 0.5, "score": 3.5, "n_refs": 1}]


def test_cases_are_bounded_and_keep_only_eligible_candidates():
    all_rows = [(SHARED, GT, [GT])] * 16
    batch = {"queries": [query(i, all_rows) for i in range(8)]}

    def scores(rows):
        result = []
        for i in range(len(rows)):
            reference = 16 if i < 16 else 0
            result.append({
                "score": 0.0, "base_reward": 0.0, "intrinsic": 0.0,
                "cot_ok": i != 0, "success": i != 1,
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


def test_ngram_eligible_stats_exclude_invalid_cots_and_empty_groups():
    batch = {"queries": [query(i, [(SHARED, GT, [GT])] * 3) for i in range(3)]}
    values = [
        (True, True, 1, 1.0), (True, True, 1, 0.0), (False, True, 1, 0.0),
        (True, False, 1, 0.0), (True, True, 0, 0.0), (False, True, 0, 0.0),
        (True, True, 1, 0.5), (True, False, 1, 0.0), (False, True, 1, 0.0),
    ]
    results = [
        {
            "score": 3.0 + reward, "base_reward": 3.0, "intrinsic": reward,
            "cot_ok": cot_ok, "success": success, "n_refs": n_refs,
        }
        for success, cot_ok, n_refs, reward in values
    ]
    report = analyze(
        [batch], {"rollout_n": 3, "n_queries": 3, "sid_length": 3},
        lambda rows: results, "ngram", case_seed=0,
    )
    assert report["summary"]["diversity"]["within_group_mean"] == pytest.approx(1 / 6)
    assert report["reward_specific"]["eligible_within_group_mean"] == 0.5
    assert report["reward_specific"]["eligible_within_group_std"] == 0.25
    assert report["reward_specific"]["eligible_queries"] == 2


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
