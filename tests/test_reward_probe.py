"""Reward 组内区分度探针(src/eval/sidreasoner_reward_probe.py)单测。

覆盖纯逻辑部分(归一化/调用/分组聚合/preset/参数解析); vLLM 生成
路径需服务器, 不在此测试。
"""

import sys
from pathlib import Path

import pytest

R4GR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R4GR_ROOT))

from src.eval.sidreasoner_reward_probe import (  # noqa: E402
    aggregate,
    call_reward,
    component_stats,
    group_rows,
    normalize_results,
    parse_kv,
    preset_kwargs,
)


def _rows():
    def r(split, idx):
        return {
            "data_source": "rec/t",
            "solution_str": (
                f"<think>\ncot for {split}{idx} {' '.join(['w'] * 30)}\n</think>\n\n<a_1><b_2><c_3>"
            ),
            "ground_truth": "<a_1><b_2><c_3>",
            "extra_info": {"split": split, "index": idx},
        }

    return [r("train", 0)] * 3 + [r("train", 1)] * 3


# --------------------------------------------------------------------------- #
# 归一化与调用
# --------------------------------------------------------------------------- #


def test_normalize_floats_and_dicts():
    assert normalize_results([0.1, 0.2], 2) == [{"score": 0.1}, {"score": 0.2}]
    assert normalize_results([{"score": 1, "td": 0.5}], 1) == [{"score": 1, "td": 0.5}]
    with pytest.raises(ValueError):
        normalize_results([0.1], 2)


def test_call_reward_batch_and_row_api():
    rows = _rows()
    batch_fn = lambda data_sources, solution_strs, ground_truths, extra_infos, **kw: [
        {"score": len(s)} for s in solution_strs
    ]
    row_fn = lambda data_source, solution_str, ground_truth, extra_info, **kw: len(solution_str)
    assert all(r["score"] > 0 for r in call_reward(batch_fn, rows))
    assert call_reward(row_fn, rows, per_row=True) == call_reward(batch_fn, rows)


# --------------------------------------------------------------------------- #
# 分组与聚合
# --------------------------------------------------------------------------- #


def test_group_rows_by_query_key():
    groups = group_rows(_rows())
    assert set(groups) == {("train", 0), ("train", 1)}
    assert all(len(v) == 3 for v in groups.values())


def test_component_stats_zero_variance_flag():
    assert component_stats([0.5, 0.5, 0.5])["zero_variance"] is True
    s = component_stats([0.1, 0.2, 0.3])
    assert s["zero_variance"] is False and s["distinct"] == 3 and s["std"] > 0


def test_aggregate_reports_within_group_discrimination():
    rows = _rows()
    # 组 0: td 全同(零方差, 无区分度); 组 1: td 各不同(有区分度)
    results = [
        {"score": 1.0 + 0.1, "base_reward": 1.0, "td": 0.5, "intrinsic": 0.1, "success": 1},
        {"score": 1.0 + 0.1, "base_reward": 1.0, "td": 0.5, "intrinsic": 0.1, "success": 1},
        {"score": 1.0 + 0.1, "base_reward": 1.0, "td": 0.5, "intrinsic": 0.1, "success": 0},
        {"score": 1.0 + 0.2, "base_reward": 1.0, "td": 0.2, "intrinsic": 0.2, "success": 1},
        {"score": 1.0 + 0.4, "base_reward": 1.0, "td": 0.4, "intrinsic": 0.4, "success": 1},
        {"score": 1.0 + 0.6, "base_reward": 1.0, "td": 0.6, "intrinsic": 0.6, "success": 1},
    ]
    report = aggregate(rows, results)
    assert report["summary"]["n_groups"] == 2
    # td: 一半组零方差
    assert report["summary"]["per_component"]["td"]["frac_groups_zero_variance"] == 0.5
    assert 0.0 < report["summary"]["per_component"]["td"]["mean_group_std"]
    # score 零方差组占比 0.5(组 0 内全同)
    assert report["summary"]["per_component"]["score"]["frac_groups_zero_variance"] == 0.5
    # 逐组 rollout 分解完整
    g0 = next(g for g in report["groups"] if g["query"]["index"] == 0)
    assert len(g0["rollouts"]) == 3
    assert {rl["td"] for rl in g0["rollouts"]} == {0.5}
    assert g0["component_stats"]["td"]["zero_variance"] is True
    # 组内 CoT 相似度: 组内文本相同 -> 1
    assert g0["within_group_cot_sim"] == pytest.approx(1.0)
    assert report["summary"]["success_rate"] == pytest.approx(5 / 6)


def test_aggregate_classifies_td_all_zero_reasons():
    """td 全零的原因分解: 无 Reference / CoT 不可解析 / 与参考雷同。"""
    rows = _rows()  # 2 组 x 3 rollouts
    results = [
        # 组 0: n_refs=0(邻居没有成功 rollout) -> no_valid_reference
        {"score": 0.0, "td": 0.0, "intrinsic": 0.0, "success": 0, "n_refs": 0, "cot_ok": 1},
        {"score": 0.0, "td": 0.0, "intrinsic": 0.0, "success": 0, "n_refs": 0, "cot_ok": 1},
        {"score": 0.0, "td": 0.0, "intrinsic": 0.0, "success": 0, "n_refs": 0, "cot_ok": 1},
        # 组 1: 有 refs 但本组 CoT 全部不可解析 -> no_usable_cot
        {"score": 0.0, "td": 0.0, "intrinsic": 0.0, "success": 0, "n_refs": 3, "cot_ok": 0},
        {"score": 0.0, "td": 0.0, "intrinsic": 0.0, "success": 0, "n_refs": 3, "cot_ok": 0},
        {"score": 0.0, "td": 0.0, "intrinsic": 0.0, "success": 0, "n_refs": 3, "cot_ok": 0},
    ]
    report = aggregate(rows, results)
    g = {x["query"]["index"]: x for x in report["groups"]}
    assert g[0]["td_all_zero_reason"] == "no_valid_reference"
    assert g[1]["td_all_zero_reason"] == "no_usable_cot"
    z = report["summary"]["groups_td_all_zero"]
    assert z == {
        "total": 2,
        "no_valid_reference": 1,
        "no_usable_cot": 1,
        "cot_identical_to_refs": 0,
    }
    assert report["summary"]["mean_n_refs"] == pytest.approx(1.5)
    assert report["summary"]["mean_cot_ok_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# preset / 参数解析
# --------------------------------------------------------------------------- #


def test_preset_kwargs_bleu_and_steprule(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET", "games")
    from config import load_config

    cfg = load_config("sidreasoner")
    entry, name, kwargs, per_row = preset_kwargs(cfg, "bleu", str(tmp_path))
    assert entry.endswith("direct_recommendation_interdiv.py") and not per_row
    assert kwargs["enable_inter_query_diversity"] is True
    assert kwargs["base_reward_type"] == "steprule"
    assert kwargs["similarity"] in ("bleu", "ngram_jaccard")
    assert int(kwargs["ngram_n"]) >= 1
    # 探针日志不得写进训练的 interdiv_metrics.jsonl
    assert kwargs["metrics_path"].startswith(str(tmp_path))

    entry2, name2, kwargs2, per_row2 = preset_kwargs(cfg, "steprule", str(tmp_path))
    assert entry2.endswith("direct_recommendation_steprule.py") and per_row2

    # ngram preset: 强制 ngram_jaccard(与 rl_ngram.sh 一致), 其余参数同 bleu
    entry3, name3, kwargs3, per_row3 = preset_kwargs(cfg, "ngram", str(tmp_path))
    assert entry3.endswith("direct_recommendation_interdiv.py") and not per_row3
    assert kwargs3["similarity"] == "ngram_jaccard"
    assert kwargs3["ngram_n"] == kwargs["ngram_n"]

    with pytest.raises(ValueError):
        preset_kwargs(cfg, "nope", str(tmp_path))


def test_parse_kv_types():
    kv = parse_kv(["a=1", "b=0.5", "c=true", "d=false", "e=path.npz"])
    assert kv == {"a": 1, "b": 0.5, "c": True, "d": False, "e": "path.npz"}
    with pytest.raises(ValueError):
        parse_kv(["bad"])
