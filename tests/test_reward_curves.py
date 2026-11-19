"""Stage 3 reward 组成曲线单测(src/eval/sidreasoner_reward_curves.py)。

伪造 logs/ 下的 interdiv JSONL(新键 + 旧字段两套), 打通文件解析、
别名回退、通配定位与出图全流程。需 matplotlib。
"""

import json
import sys
from pathlib import Path

import pytest

R4GR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R4GR_ROOT))

from src.eval.sidreasoner_reward_curves import (  # noqa: E402
    KEY_ALIASES,
    SUPPORTED_KEYS,
    load_metrics,
    plot_reward_curves,
    resolve_metrics_path,
    series_points,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture()
def fake_logs(tmp_path):
    """伪造两个实验的指标 JSONL: _ngram 用新键, _bleu 只用旧字段。"""
    logs = tmp_path / "logs"

    def row(step, mean, mx, mn, new_keys):
        rec = {"step": step, "lambda": 0.2}
        if new_keys:
            rec.update({
                "critic/correctness_rewards/mean": mean,
                "critic/correctness_rewards/max": mx,
                "critic/correctness_rewards/min": mn,
                "critic/diversity_rewards/mean": mean / 10,
                "critic/diversity_rewards/max": mx / 10,
                "critic/diversity_rewards/min": 0.0,
                "critic/total_rewards/mean": mean * 1.1,
                "critic/total_rewards/max": mx * 1.1,
                "critic/total_rewards/min": mn,
            })
        else:
            rec.update({
                "base_reward_mean": mean,
                "intrinsic_mean": mean / 10,
                "augmented_reward_mean": mean * 1.1,
            })
        return rec

    _write_jsonl(
        logs / "Video_Games_stage3_rl_Qwen3-1.7B_ngram_interdiv_metrics.jsonl",
        [row(s, 0.1 + 0.01 * s, 1.1, 0.0, True) for s in (1, 2, 3, 4, 5)],
    )
    # 旧字段文件 + 带 _own SID 后缀的文件(验证通配)
    _write_jsonl(
        logs / "Video_Games_stage3_rl_Qwen3-1.7B_bleu_interdiv_metrics.jsonl",
        [row(s, 0.2 + 0.01 * s, 1.1, 0.0, False) for s in (1, 2, 3)],
    )
    return logs


def test_resolve_metrics_path_suffix_and_own(fake_logs):
    got = resolve_metrics_path("Video_Games", "_ngram", str(fake_logs))
    assert got and got.endswith("Qwen3-1.7B_ngram_interdiv_metrics.jsonl")
    assert resolve_metrics_path("Video_Games", "_nope", str(fake_logs)) is None


def test_load_metrics_sorts_and_skips_bad_lines(fake_logs):
    p = fake_logs / "tmp_mixed.jsonl"
    _write_jsonl(p, [{"step": 3, "x": 1}, "not json", {"step": 1, "x": 0}])
    recs = load_metrics(str(p))
    assert [r["step"] for r in recs] == [1, 3]


def test_series_points_new_keys_and_legacy_alias(fake_logs):
    ngram = load_metrics(str(
        fake_logs / "Video_Games_stage3_rl_Qwen3-1.7B_ngram_interdiv_metrics.jsonl"
    ))
    pts = series_points(ngram, "critic/correctness_rewards/max")
    assert [v for _, v in pts] == [1.1] * 5
    # 旧字段文件: mean 经别名回退可得, max 无别名 -> 空序列
    bleu = load_metrics(str(
        fake_logs / "Video_Games_stage3_rl_Qwen3-1.7B_bleu_interdiv_metrics.jsonl"
    ))
    assert len(series_points(bleu, "critic/correctness_rewards/mean")) == 3
    assert series_points(bleu, "critic/correctness_rewards/max") == []
    with pytest.raises(ValueError):
        series_points(bleu, "bogus/key")


def test_supported_keys_complete():
    assert len(SUPPORTED_KEYS) == 9
    assert set(KEY_ALIASES) == {
        "critic/correctness_rewards/mean",
        "critic/diversity_rewards/mean",
        "critic/total_rewards/mean",
    }


def test_plot_reward_curves_end_to_end(fake_logs, tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET", "games")
    out = tmp_path / "rewards.png"
    series = [
        {"label": "Correctness (n-gram)", "exp": "_ngram",
         "key": "critic/correctness_rewards/mean"},
        {"label": "Diversity (n-gram)", "exp": "_ngram",
         "key": "critic/diversity_rewards/mean"},
        {"label": "Total (n-gram)", "exp": "_ngram",
         "key": "critic/total_rewards/mean"},
        # 旧字段 JSONL: 走别名回退
        {"label": "Correctness (BLEU, legacy)", "exp": "_bleu",
         "key": "critic/correctness_rewards/mean"},
    ]
    got = plot_reward_curves(series=series, out_path=str(out),
                             logs_root=str(fake_logs))
    assert got == str(out) and out.exists() and out.stat().st_size > 0


def test_plot_reward_curves_no_data_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET", "games")
    with pytest.raises(SystemExit):
        plot_reward_curves(series=[{"label": "x", "exp": "_none",
                                    "key": "critic/total_rewards/mean"}],
                           out_path=str(tmp_path / "none.png"),
                           logs_root=str(tmp_path))
