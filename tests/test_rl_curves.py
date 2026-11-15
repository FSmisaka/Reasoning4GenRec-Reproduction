"""RL 训练曲线绘图单测(src/eval/sidreasoner_rl_curves.py)。

本地没有真实评测结果, 用伪造的 step json(结构与评测产物一致:
贪心=逐样本 list[dict](predict/output), 采样=summary dict)打通全流程。
需 matplotlib(系统 python3 自带)。
"""

import json
import sys
from pathlib import Path

import pytest

R4GR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R4GR_ROOT))


GT = "<a_1><b_1><c_1>"
FILLERS = [f"<a_9><b_9><c_{k}>" for k in range(10)]


def write_greedy(path):
    """伪造贪心评测结果: 4 样本 x 10 候选束, 一半首位命中(HR@K=NDCG@K=0.5)。"""
    data = []
    for i in range(4):
        hit = i < 2
        # 命中: 目标在首位; 未命中: 10 个候选全不含目标
        preds = ([GT] + FILLERS[:9]) if hit else FILLERS[:10]
        data.append({"output": GT, "predict": preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_summary(path, mean, half=0.02):
    """伪造采样评测 summary(aggregate.metrics 含 mean/95% CI)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "aggregate": {
            "metrics": {
                "NDCG@10": {"mean": mean, "std": half / 2,
                            "ci95_lower": mean - half, "ci95_upper": mean + half},
                "NDCG@5": {"mean": mean, "std": half / 2,
                           "ci95_lower": mean - half, "ci95_upper": mean + half},
                "HR@10": {"mean": mean, "std": half / 2,
                          "ci95_lower": mean - half, "ci95_upper": mean + half},
                "HR@5": {"mean": mean, "std": half / 2,
                         "ci95_lower": mean - half, "ci95_upper": mean + half},
            }
        }
    }
    path.write_text(json.dumps(data))


@pytest.fixture()
def fake_results(tmp_path, monkeypatch):
    """伪造 results/sidreasoner/Games/{三个变体目录} + info 文件 + config 数据源。"""
    root = tmp_path / "results" / "sidreasoner"
    games = root / "Games"

    # rolloutbeam: 100-500 全有; rolloutbeam_div: 只有 100-300(验证截断);
    # steprule(_temp) 与 rolloutbeam_temp: 用 summary(采样评测) 验证 CI 路径
    for step in (100, 200, 300, 400, 500):
        write_greedy(games / "think_rolloutbeam" / f"step_{step}.json")
    for step in (100, 200, 300):
        write_greedy(games / "think_rolloutbeam_div" / f"step_{step}.json")
    for step in (100, 200, 300, 400, 500):
        write_summary(games / "think_steprule_temp" / f"step_{step}_summary.json",
                      mean=0.4 + step / 100 * 0.01)
    for step in (100, 200, 300):
        write_summary(games / "think_rolloutbeam_temp" / f"step_{step}_summary.json",
                      mean=0.3 + step / 100 * 0.02)

    # info 文件 + 数据根目录(挂到 config 的 data.root)
    data_root = tmp_path / "data"
    (data_root / "info").mkdir(parents=True)
    (data_root / "info" / "Video_Games_5_2016-10-2018-11.txt").write_text(
        GT + "\tHitItem\t1\n" + "".join(f"{f}\tFiller{i}\t{i+2}\n" for i, f in enumerate(FILLERS))
    )

    import config as config_mod

    real_load = config_mod.load_config

    def patched_load(name):
        cfg = real_load(name)
        cfg["data"]["root"] = str(data_root)
        return cfg

    monkeypatch.setattr(config_mod, "load_config", patched_load)
    # rl_curves 里 `from config import load_config` 拿到的是同一函数对象
    import src.eval.sidreasoner_rl_curves as curves

    monkeypatch.setattr(curves, "load_config", patched_load)
    return root


def test_normalize_metric_aliases():
    import src.eval.sidreasoner_rl_curves as curves

    assert curves.normalize_metric("NDCG@10") == "NDCG@10"
    assert curves.normalize_metric("ndcg@5") == "NDCG@5"
    assert curves.normalize_metric("Recall@10") == "HR@10"
    assert curves.normalize_metric("HitRate@5") == "HR@5"
    assert curves.normalize_metric("Recall(HitRate)@10") == "HR@10"
    with pytest.raises(ValueError):
        curves.normalize_metric("MRR@10")


def test_series_points_greedy_and_summary(fake_results):
    import src.eval.sidreasoner_rl_curves as curves
    from src.eval.sidreasoner_metrics import load_item_names

    item_names = load_item_names(
        str(fake_results.parent.parent / "data" / "info" / "Video_Games_5_2016-10-2018-11")
    )
    # 贪心: 4 样本命中 2 -> HR@5 = HR@10 = 0.5; 命中在首位
    pts = curves.series_points(str(fake_results / "Games" / "think_rolloutbeam"),
                               item_names, "HR@10")
    assert [p[0] for p in pts] == [100, 200, 300, 400, 500]
    assert all(abs(p[1] - 0.5) < 1e-6 for p in pts)
    assert all(p[2] is None and p[3] is None for p in pts)

    # summary: 取 aggregate mean + 95% CI(ci95_lower/ci95_upper)
    pts = curves.series_points(str(fake_results / "Games" / "think_steprule_temp"),
                               item_names, "NDCG@10")
    assert [p[0] for p in pts] == [100, 200, 300, 400, 500]
    assert pts[0][1] == pytest.approx(0.4 + 0.01)
    assert pts[0][2] == pytest.approx(pts[0][1] - 0.02)
    assert pts[0][3] == pytest.approx(pts[0][1] + 0.02)


def test_paper_baseline_table():
    """官方论文 Table 2 参考值(SIDReasoner 思考模式最终结果)。"""
    import src.eval.sidreasoner_rl_curves as curves

    b = curves.PAPER_BASELINE
    assert b["Games"]["HR@5"] == 0.0710 and b["Games"]["NDCG@10"] == 0.0563
    assert b["Office"]["NDCG@5"] == 0.1119 and b["Office"]["HR@10"] == 0.1648
    assert b["Industrial"]["HR@5"] == 0.1109 and b["Industrial"]["NDCG@10"] == 0.1010
    assert set(b["Games"]) == {"HR@5", "NDCG@5", "HR@10", "NDCG@10"}


def test_plot_rl_metric_end_to_end(fake_results, capsys):
    import src.eval.sidreasoner_rl_curves as curves

    out = curves.plot_rl_metric("NDCG@10", dataset="games",
                                root=str(fake_results))
    assert out.endswith("Games/figures/NDCG@10.png")
    assert (fake_results / "Games" / "figures" / "NDCG@10.png").stat().st_size > 10000
    # 论文参考线分支确实执行(Games NDCG@10 = 0.0563)
    assert "[baseline] Games NDCG@10 = 0.0563" in capsys.readouterr().out

    # 可用 paper_baseline=False 关闭参考线
    curves.plot_rl_metric("NDCG@10", dataset="games", root=str(fake_results),
                          paper_baseline=False)
    assert "[baseline]" not in capsys.readouterr().out

    # 别名 + HR 指标
    out = curves.plot_rl_metric("Recall(HitRate)@10", dataset="games",
                                root=str(fake_results))
    assert (fake_results / "Games" / "figures" / "HR@10.png").exists()

    # 扩展性: 追加一条自定义折线(不存在的目录 -> 自动跳过不出线)
    out = curves.plot_rl_metric(
        "HR@5", dataset="games", root=str(fake_results),
        series=curves.DEFAULT_SERIES
        + [{"label": "Future", "subdir": "think_future"}],
    )
    assert (fake_results / "Games" / "figures" / "HR@5.png").exists()


def test_plot_rl_metric_empty_fails(fake_results):
    import src.eval.sidreasoner_rl_curves as curves

    with pytest.raises(SystemExit):
        curves.plot_rl_metric(
            "NDCG@10", dataset="games", root=str(fake_results),
            series=[{"label": "nowhere", "subdir": "think_missing"}],
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
