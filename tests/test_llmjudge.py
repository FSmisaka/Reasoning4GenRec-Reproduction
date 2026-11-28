"""LLM-Judge CoT 相似度 scorer 单测(不依赖网络 / vLLM 服务)。"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R4GR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R4GR_ROOT))

from src.reward.interdiv.llmjudge import (  # noqa: E402
    BLOCK_SEP,
    HistoryLookup,
    LLMJudge,
    parse_scores,
    render_block,
    split_block,
)
from src.reward.interdiv.reward import InterQueryDiversityAddon  # noqa: E402

K3 = ("evidence", "strategy", "profile")
GT = "<a_1><b_2><c_3>"


def _nested(scores):
    return json.dumps(
        {k: {"score": v, "quote_a": "qa", "quote_b": "qb"} for k, v in zip(K3, scores)}
    )


def test_parse_scores_nested_fenced_flat_clamped():
    assert parse_scores(_nested((3, 2, 4))) == dict(zip(K3, (3, 2, 4)))
    assert parse_scores(f"Scores:\n```json\n{_nested((1, 2, 3))}\n```")["profile"] == 3
    # 扁平整数写法也接受
    assert parse_scores(json.dumps(dict(zip(K3, (2, 3, 4))))) == dict(zip(K3, (2, 3, 4)))
    # 越界裁剪; 缺键 / 非 JSON -> None
    out = parse_scores(_nested((9, -1, 2)))
    assert out == dict(zip(K3, (4, 0, 2)))
    assert parse_scores('{"evidence": 2}') is None
    assert parse_scores('{"valid": false}') is None
    assert parse_scores("no json here") is None


def test_parse_scores_regex_fallback():
    # note 复读把 JSON 撑爆, 分数在坏掉之前
    note_loop = (
        '{"evidence": 4, "strategy": 3, "profile": 3, "note": "Both traces infer '
        "a preference for racing games, and the user is a racing fan, and the user"
    )
    assert parse_scores(note_loop) == dict(zip(K3, (4, 3, 3)))
    # 无引号键 / 多余引号冒号
    assert parse_scores("{ evidence: 4, strategy: 3, profile: 3 }") == dict(zip(K3, (4, 3, 3)))
    assert parse_scores('{"evidence":": 4, "strategy": 3, "profile": 3') == dict(
        zip(K3, (4, 3, 3))
    )
    # 某键的值退化为文本(无数字) -> 整体回退 None
    assert parse_scores('{"evidence": the the the, "strategy": 3, "profile": 3}') is None


def test_block_round_trip():
    lines = ["<a_1><b_1><c_1> - Game One", "<a_2><b_2><c_2> - Game Two"]
    block = render_block(lines, "the user likes racing")
    history, cot = split_block(block)
    assert history.splitlines() == lines
    assert cot == "the user likes racing"
    assert BLOCK_SEP in block


def test_prompt_template_placeholders():
    judge = LLMJudge()
    for p in ("{{history_a}}", "{{cot_a}}", "{{history_b}}", "{{cot_b}}"):
        assert p in judge.template


def test_history_lookup_from_csv(tmp_path):
    df = pd.DataFrame(
        [
            {
                "history_item_sid": "['<a_1><b_2><c_3>', '<a_4><b_5><c_6>']",
                "history_item_title": "['Game One', 'Game Two']",
            }
        ]
    )
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    df.to_csv(split_dir / "X_5_2016-10-2018-11.csv", index=False)
    lookup = HistoryLookup(str(tmp_path), "X")
    assert lookup.lines("train", 0) == [
        "<a_1><b_2><c_3> - Game One",
        "<a_4><b_5><c_6> - Game Two",
    ]


def test_judge_similarity_and_diversity_from_scores():
    judge = LLMJudge()
    a = render_block(["<a_1><b_1><c_1> - A"], "cot a")
    b = render_block(["<a_2><b_2><c_2> - B"], "cot b")
    judge.judge_pairs = lambda pairs: [{k: 4 for k in K3} for _ in pairs]
    assert judge.similarity(a, b) == pytest.approx(1.0)
    assert judge.diversity(a, b) == pytest.approx(0.0)
    judge.judge_pairs = lambda pairs: [{k: 1 for k in K3} for _ in pairs]
    assert judge.diversity(a, b) == pytest.approx(0.75)
    assert judge.diversities(a, [b, b]) == pytest.approx([0.75, 0.75])


def _resp(cot, answer):
    return f"<think>\n{cot}\n</think>\n\n{answer}"


@pytest.mark.parametrize(("mode", "expected"), [("none", [0.3, 0.6]), ("batch_minmax", [0.0, 1.0])])
def test_addon_llmjudge_uses_history_blocks(tmp_path, monkeypatch, mode, expected):
    n_q, g = 2, 2
    emb = np.eye(n_q, 8, dtype=np.float32)
    path = tmp_path / "user_emb.npz"
    np.savez(
        path,
        split=np.asarray(["train"] * n_q),
        index=np.arange(n_q, dtype=np.int64),
        embeddings=emb,
        history_items=np.asarray([np.arange(3), np.arange(3, 6)], dtype=object),
        meta=np.asarray(
            json.dumps({"category": "X", "data_dir": str(tmp_path / "rec_reasoning_verl" / "X")})
        ),
    )

    captured = {}

    class FakeJudge:
        def __init__(self, *a, **kw):
            self.n_calls = 0
            self.n_fallback = 0

        def similarities(self, pairs):
            captured["pairs"] = pairs
            return [0.7 if "reasoning about user 0" in a else 0.4 for a, _ in pairs]

    class FakeHistory:
        def __init__(self, *a, **kw):
            pass

        def lines(self, split, index):
            return [f"<a_{index}><b_0><c_0> - item {index}"]

    monkeypatch.setattr("src.reward.interdiv.reward.LLMJudge", FakeJudge)
    monkeypatch.setattr("src.reward.interdiv.reward.HistoryLookup", FakeHistory)

    addon = InterQueryDiversityAddon(
        user_embedding_path=str(path),
        similarity="llmjudge",
        top_k_neighbors=1,
        lambda_init=1.0,
        sigma=1.0,
        td_norm=mode,
        metrics_path=str(tmp_path / "metrics.jsonl"),
    )
    rows = []
    for qi in range(n_q):
        for gi in range(g):
            ok = gi == 0
            rows.append(
                {
                    "data_source": "rec/test",
                    "solution_str": _resp(f"reasoning about user {qi} take {gi}", GT if ok else "<a_9><b_9><c_9>"),
                    "ground_truth": GT,
                    "extra_info": {"split": "train", "index": qi},
                }
            )
    out = addon.compute_batch(rows, [0.5] * len(rows))

    pairs = captured["pairs"]
    assert pairs and all(BLOCK_SEP in a and BLOCK_SEP in b for a, b in pairs)
    assert any("reasoning about user" in a for a, _ in pairs)
    assert all("- item " in b for _, b in pairs)
    assert out[0]["n_refs"] == 1
    assert out[0]["td"] == pytest.approx(0.3)
    assert [out[i]["intrinsic"] for i in (0, 2)] == pytest.approx(expected)
    assert out[1]["intrinsic"] == 0.0  # 失败 rollout 不发放
