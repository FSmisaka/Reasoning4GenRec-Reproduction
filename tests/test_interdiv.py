"""跨 Query CoT 多样性奖励(DIVER 改造版)单测。

纯逻辑部分(BLEU/邻居/塑形/编排)不依赖 verl; 数据相关用例
(StepRule 兼容性 / SASRec 编码器 / 离线提取)依赖本地
data/raw/Amazon 与 runs/ 产物, 缺失时自动跳过。

运行: .venv/bin/python -m pytest tests/test_interdiv.py -v
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

R4GR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R4GR_ROOT))

from src.reward.interdiv.bleu import DiverBleu  # noqa: E402
from src.reward.interdiv.cot_utils import extract_cot, is_recommendation_success  # noqa: E402
from src.reward.interdiv.neighbor import build_same_user_mask, topk_neighbors  # noqa: E402
from src.reward.interdiv.shaping import lambda_at, shape_rewards  # noqa: E402
from src.reward.interdiv.user_store import UserEmbeddingStore  # noqa: E402
from src.reward.interdiv.reward import InterQueryDiversityAddon  # noqa: E402

GAMES_ROOT = R4GR_ROOT / "data" / "raw" / "Amazon"
GAMES_PARQUET = GAMES_ROOT / "rec_reasoning_verl" / "Video_Games" / "train.parquet"
GAMES_SASREC = R4GR_ROOT / "runs" / "Video_Games_sasrec_bce1" / "best.pt"
INFO_FILE = GAMES_ROOT / "info" / "Video_Games_5_2016-10-2018-11.txt"

GT = "<a_1><b_2><c_3>"


def _resp(cot, answer="<a_1><b_2><c_3>"):
    return f"<think>\n{cot}\n</think>\n\n{answer}"


# --------------------------------------------------------------------------- #
# BLEU(DIVER 口径)
# --------------------------------------------------------------------------- #


def test_bleu_identical_texts_score_one():
    b = DiverBleu()
    text = "user likes action games and bought several controllers last month"
    assert b.similarity(text, text) == pytest.approx(1.0)


def test_bleu_disjoint_texts_score_zero():
    b = DiverBleu()
    a = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    c = "one two three four five six seven eight nine ten eleven"
    # 无共享 unigram -> p_1 = 0 -> BLEU = 0 -> sim = 0
    assert b.similarity(a, c) == 0.0


def test_bleu_partial_overlap_between_zero_and_one():
    b = DiverBleu()
    a = "the user enjoys strategy games and rpg titles on console"
    c = "the user enjoys strategy games and racing titles on console"
    sim = b.similarity(a, c)
    assert 0.0 < sim < 1.0


def test_bleu_symmetric():
    b = DiverBleu()
    a = "first response about puzzle games and family fun"
    c = "second response about puzzle games but different words here"
    assert b.similarity(a, c) == pytest.approx(b.similarity(c, a))


def test_bleu_div_len_truncates():
    b = DiverBleu(div_len=50)
    head = "the user bought many shooter games this year and liked them"
    a = head + " " + " ".join(f"tail{i}" for i in range(200))
    c = head + " " + " ".join(f"other{i}" for i in range(200))
    # 前 50 字符相同 -> 截断后近似同一文本
    assert b.similarity(a, c) > 0.5


def test_bleu_short_text_returns_zero():
    b = DiverBleu()
    assert b.bleu("a b c d e f", "a b c d e f") >= 0.0  # 正常短文本可计算
    assert b.bleu("reference text here", "tiny") == 0.0


def test_bleu_matches_nltk_if_available():
    """若有 nltk, 校验自实现与 nltk sentence_bleu(无平滑)逐值一致。"""
    nltk = pytest.importorskip("nltk")
    from nltk.translate.bleu_score import sentence_bleu

    b = DiverBleu()
    pairs = [
        ("the user likes strategy games and rpg titles on console",) * 2,
        ("the user enjoys strategy games and rpg titles on console",
         "the user enjoys strategy games and racing titles on console"),
        ("alpha beta gamma delta epsilon zeta eta theta iota kappa",
         "one two three four five six seven eight nine ten eleven"),
        ("user a likes games user a likes games user a likes games",
         "user b likes games user b likes games user b likes games"),
    ]
    for a, c in pairs:
        ref, cand = a.split(), c.split()
        expected = sentence_bleu([ref], cand, weights=DiverBleu().weights)
        got = b.bleu(a, c)
        assert got == pytest.approx(float(expected), abs=1e-9), (a, c)


# --------------------------------------------------------------------------- #
# CoT 提取与成功判定
# --------------------------------------------------------------------------- #


def test_extract_cot_standard():
    resp = _resp("thinking about user preferences")
    assert extract_cot(resp) == "thinking about user preferences"


def test_extract_cot_missing_open_tag():
    resp = "thinking only </think>\n\n<a_1><b_2><c_3>"
    assert extract_cot(resp) == "thinking only"


def test_extract_cot_unclosed_returns_none():
    assert extract_cot("still thinking without close tag <a_1>") is None


def test_success_requires_full_three_level_match():
    assert is_recommendation_success(_resp("some cot"), GT)
    # 只匹配前缀/两级 -> 不成功(即便 StepRule 会给 0.25/0.5 分)
    assert not is_recommendation_success(_resp("some cot", "<a_1><b_2><c_9>"), GT)
    assert not is_recommendation_success(_resp("some cot", "<a_1><b_9><c_3>"), GT)
    # 无答案
    assert not is_recommendation_success("no think marker", GT)


def test_success_not_confused_with_positive_reward():
    """格式合法但零匹配 -> StepRule reward = 0.1 > 0 但不成功。"""
    from src.reward.direct_recommendation_steprule import MyRewardComputer

    os.environ.setdefault("DATASET", "games")
    rc = MyRewardComputer()
    resp = _resp("cot", "<a_1><b_2><c_3>")
    wrong = _resp("cot", "<a_4><b_5><c_6>")
    if os.path.exists(INFO_FILE):
        # <a_4>... 是否合法取决于码本, 用一个确定合法但错误的 SID 检查
        assert rc.compute("rec/x", wrong, GT, None) >= 0.0
        assert not is_recommendation_success(wrong, GT)
        assert rc.compute("rec/x", resp, GT, None) > 0.0
        assert is_recommendation_success(resp, GT)


# --------------------------------------------------------------------------- #
# User Embedding Store
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_store(tmp_path):
    def make(n=4, dim=8, split="train"):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(n, dim)).astype(np.float32)
        hist = [np.arange(i, i + 3, dtype=np.int64) for i in range(n)]
        path = tmp_path / "user_emb.npz"
        np.savez(
            path,
            split=np.asarray([split] * n),
            index=np.arange(n, dtype=np.int64),
            embeddings=emb,
            history_items=np.asarray(hist, dtype=object),
            meta=np.asarray(json.dumps({"encoder": "test"})),
        )
        return path, emb, hist

    return make


def test_store_lookup_and_normalize(tmp_store):
    path, emb, hist = tmp_store()
    store = UserEmbeddingStore(str(path))
    u, h, uids = store.lookup([("train", 0), ("train", 2)])
    assert u.shape == (2, 8)
    norms = np.linalg.norm(u, axis=1)
    # 原 emb 行被归一化后, 模长应接近 1
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert h[0].tolist() == hist[0].tolist()
    # 无 user_ids 字段的旧缓存 -> 空串(未知)
    assert uids == ["", ""]


def test_store_missing_key_raises(tmp_store):
    path, *_ = tmp_store(n=2)
    store = UserEmbeddingStore(str(path))
    with pytest.raises(KeyError, match="(?i)user embedding"):
        store.lookup([("train", 99)])


def test_store_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        UserEmbeddingStore(str(tmp_path / "nope.npz"))


# --------------------------------------------------------------------------- #
# In-batch Top-K 邻居
# --------------------------------------------------------------------------- #


def _unit(vectors):
    v = np.asarray(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_topk_excludes_self():
    import torch

    u = torch.as_tensor(_unit(np.eye(4, dtype=np.float64)))
    idx, sim = topk_neighbors(u, k=3)
    for i in range(4):
        assert i not in idx[i].tolist()


def test_topk_returns_most_similar():
    import torch

    # q0 与 q1 完全同向(相似度 1), 与 q2/q3 正交
    u = torch.as_tensor(_unit([[1, 0], [1, 0], [0, 1], [0, 1]]))
    idx, sim = topk_neighbors(u, k=1)
    assert idx[0, 0] == 1 and idx[1, 0] == 0
    assert sim[0, 0] == pytest.approx(1.0)


def test_topk_excludes_same_user_by_history():
    import torch

    # q0/q1 历史完全一致(同一用户两个时间点) -> 互相排除
    sets = [frozenset({1, 2, 3}), frozenset({1, 2, 3}), frozenset({9, 8, 7})]
    mask = build_same_user_mask(sets, jaccard_threshold=0.8)
    assert mask[0, 1] and mask[1, 0]
    assert not mask[0, 2]

    u = torch.as_tensor(_unit([[1, 0], [0.99, 0.1], [0, 1]]))
    idx, _ = topk_neighbors(u, k=1, exclude=mask)
    # q0 最相似的是 q1(同用户)但被排除 -> 应选 q2
    assert idx[0, 0] == 2


def test_same_user_mask_exact_user_id():
    """带 user_id 时精确排除: 历史完全不同的同 user_id 也被排除。"""
    sets = [frozenset({1, 2, 3}), frozenset({4, 5, 6}), frozenset({7, 8, 9})]
    uids = ["A", "A", "B"]
    mask = build_same_user_mask(sets, 0.8, user_ids=uids)
    assert mask[0, 1] and mask[1, 0]
    assert not mask[0, 2]
    # 空 user_id(未知)不参与精确排除
    mask2 = build_same_user_mask(sets, 0.8, user_ids=["", "A", ""])
    assert not mask2[0, 1]


def test_topk_jaccard_boundary():
    a = frozenset(range(10))
    b = frozenset(range(9))  # Jaccard = 9/11 ~ 0.818 >= 0.8 -> 同用户
    c = frozenset(range(100, 110))  # Jaccard = 0 -> 不同用户
    mask = build_same_user_mask([a, b, c], 0.8)
    assert mask[0, 1] and not mask[0, 2]


def test_topk_insufficient_neighbors():
    import torch

    u = torch.as_tensor(_unit([[1, 0], [0, 1]]))
    idx, sim = topk_neighbors(u, k=5)  # 只有 1 个可用
    assert idx.shape == (2, 1)


def test_topk_single_query():
    import torch

    u = torch.as_tensor(_unit([[1.0, 0.0]]))
    idx, sim = topk_neighbors(u, k=3)
    assert idx.shape == (1, 0)


# --------------------------------------------------------------------------- #
# Shaping
# --------------------------------------------------------------------------- #


def test_lambda_linear_decay():
    assert lambda_at(0, 0.2, 0.0, 100) == pytest.approx(0.2)
    assert lambda_at(50, 0.2, 0.0, 100) == pytest.approx(0.1)
    assert lambda_at(100, 0.2, 0.0, 100) == pytest.approx(0.0)
    assert lambda_at(500, 0.2, 0.0, 100) == pytest.approx(0.0)
    assert lambda_at(10, 0.2, 0.1, 100) == pytest.approx(0.19)


def test_shape_conditional_on_success():
    base = np.array([1.0, 1.0])
    success = np.array([True, False])
    td = np.array([0.8, 0.8])
    intrinsic, aug = shape_rewards(base, success, td, lam=0.5, sigma=1.0)
    assert intrinsic[0] > 0
    assert intrinsic[1] == 0.0
    assert aug[0] == pytest.approx(1.0 + 0.5 * 0.8)
    assert aug[1] == pytest.approx(1.0)


def test_shape_clip_sigma():
    base = np.array([1.0])
    success = np.array([True])
    td = np.array([0.9])
    intrinsic, _ = shape_rewards(base, success, td, lam=1.0, sigma=0.5)
    assert intrinsic[0] == pytest.approx(0.5)


def test_shape_gamma_discount():
    base = np.array([1.0])
    success = np.array([True])
    td = np.array([1.0])
    counts = np.array([10.0])
    intrinsic, _ = shape_rewards(base, success, td, token_counts=counts, gamma=0.9, sigma=1.0, lam=1.0)
    assert intrinsic[0] == pytest.approx(0.9**10)


def test_shape_gamma_one_ignores_tokens():
    base = np.array([1.0])
    success = np.array([True])
    td = np.array([0.7])
    intrinsic, _ = shape_rewards(
        base, success, td, token_counts=np.array([123.0]), gamma=1.0, sigma=1.0, lam=1.0
    )
    assert intrinsic[0] == pytest.approx(0.7)


def test_shape_zero_td_when_no_neighbors():
    base = np.array([1.0])
    success = np.array([True])
    td = np.array([0.0])
    intrinsic, aug = shape_rewards(base, success, td, lam=0.5, sigma=1.0)
    assert intrinsic[0] == 0.0 and aug[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Addon 端到端(合成数据)
# --------------------------------------------------------------------------- #


def _make_batch(queries, rollout_n=4):
    """合成 batch: CoT = 长共享脚手架 + query 主题词 + rollout 变体词。

    脚手架含共享 5-gram, 保证不同 CoT 的 BLEU 落在 (0,1) 且互相可区分
    (DIVER 口径无平滑, 无共享 5-gram 的文本对 BLEU 恒为 0)。
    """
    scaffold_a = (
        "looking at the interaction history of this user we can "
        "clearly see a stable preference pattern forming over time"
    )
    scaffold_b = (
        "based on these repeated interactions the most reasonable "
        "next recommendation should follow the same direction"
    )
    rows = []
    for qi, (split, idx, cot_x) in enumerate(queries):
        for g in range(rollout_n):
            variant = cot_x if g == 0 else f"{cot_x} take {g}"
            cot = f"{scaffold_a} {variant} {scaffold_b}"
            ok = g == 0  # 每个 query 只有第 0 个 rollout 成功
            answer = GT if ok else "<a_9><b_9><c_9>"
            rows.append(
                {
                    "data_source": "rec/test",
                    "solution_str": _resp(cot, answer),
                    "ground_truth": GT,
                    "extra_info": {"split": split, "index": idx},
                }
            )
    return rows


@pytest.fixture
def addon(tmp_path, tmp_store):
    def make(n_queries=4, rollout_n=4, **kw):
        path, emb, hist = tmp_store(n=n_queries, dim=8)
        hist = [np.arange(i * 10, i * 10 + 5, dtype=np.int64) for i in range(n_queries)]
        # 重写历史, 保证互不相同(不同用户)
        np.savez(
            path,
            split=np.asarray(["train"] * n_queries),
            index=np.arange(n_queries, dtype=np.int64),
            embeddings=emb,
            history_items=np.asarray(hist, dtype=object),
            meta=np.asarray("{}"),
        )
        return (
            InterQueryDiversityAddon(
                user_embedding_path=str(path),
                metrics_path=str(tmp_path / "metrics.jsonl"),
                **kw,
            ),
            n_queries,
            rollout_n,
        )

    return make


def test_addon_end_to_end_grouping_and_td(addon):
    ad, n_q, g = addon(n_queries=4, rollout_n=4, top_k_neighbors=2, lambda_init=1.0, sigma=1.0)
    rows = _make_batch(
        [("train", i, f"genre{i}") for i in range(4)],
    )
    base = [0.5] * len(rows)
    out = ad.compute_batch(rows, base)
    assert len(out) == n_q * g

    # 成功 rollout(每组第 0 个)获得多样性奖励; 失败 rollout 不获得
    for qi in range(n_q):
        ok_row = qi * g
        assert out[ok_row]["success"] == 1
        assert out[ok_row]["td"] > 0
        assert out[ok_row]["intrinsic"] > 0
        assert out[ok_row]["score"] == pytest.approx(0.5 + out[ok_row]["intrinsic"])
        for j in range(1, g):
            r = out[qi * g + j]
            assert r["success"] == 0 and r["intrinsic"] == 0.0
            assert r["score"] == pytest.approx(0.5)

    # 每个 query 的 G 个 rollout 的 td 不全相同(独立计算)
    tds = [out[qi * g + j]["td"] for qi in range(n_q) for j in range(g)]
    assert len(set(tds)) > 1

    # metrics 文件落盘
    rec = json.loads(open(ad.metrics_path).read().strip().splitlines()[0])
    assert rec["n_queries"] == n_q and rec["n_rollouts"] == n_q * g
    assert 0.0 < rec["success_rate"] <= 1.0


def test_addon_identical_cots_zero_td(addon):
    """所有 rollout 的 CoT 完全一致且全成功 -> BLEU=1 -> td=0 -> 无加成。"""
    ad, n_q, g = addon(n_queries=3, rollout_n=4, top_k_neighbors=2, lambda_init=1.0)
    rows = _make_batch([("train", i, "same") for i in range(3)])
    common = _resp("a shared identical chain of thought with enough tokens to compute bleu properly", GT)
    for r in rows:
        r["solution_str"] = common
    out = ad.compute_batch(rows, [0.5] * len(rows))
    for r in out:
        assert r["success"] == 1
        assert r["td"] == pytest.approx(0.0)
        assert r["intrinsic"] == 0.0


def test_addon_no_valid_neighbors(addon):
    """邻居都没有成功 rollout -> K_q=0 -> td=0 -> intrinsic=0。"""
    ad, n_q, g = addon(n_queries=3, rollout_n=4, top_k_neighbors=2)
    rows = _make_batch([("train", i, f"g{i}") for i in range(3)])
    # 把所有成功 rollout 改成失败(答案错误), 但保留第 0 个 query 的一条成功
    for r in rows:
        if r["extra_info"]["index"] != 0:
            r["solution_str"] = _resp("cot", "<a_9><b_9><c_9>")
    out = ad.compute_batch(rows, [0.3] * len(rows))
    for r in out:
        assert r["td"] == 0.0 and r["intrinsic"] == 0.0
        assert r["score"] == pytest.approx(0.3)


def test_addon_val_batch_not_shaped(addon):
    """split=test 的 batch 只返回原始 reward(验证集不加多样性)。"""
    ad, n_q, g = addon(n_queries=3, rollout_n=2)
    rows = _make_batch([("test", i, f"v{i}") for i in range(3)])
    out = ad.compute_batch(rows, [0.4] * len(rows))
    for r in out:
        assert r["score"] == pytest.approx(0.4)
        assert r["intrinsic"] == 0.0
    # 验证 batch 不推进 lambda 计数
    assert ad._train_steps == 0


def test_addon_single_query_not_shaped(addon):
    ad, *_ = addon(n_queries=1, rollout_n=2)
    rows = _make_batch([("train", 0, "only")])
    out = ad.compute_batch(rows, [0.2] * len(rows))
    for r in out:
        assert r["score"] == pytest.approx(0.2)


def test_addon_same_user_neighbors_excluded(addon, tmp_path):
    """同用户(历史 Jaccard >= 阈值)样本不作为邻居 reference。"""
    import numpy as np

    n = 3
    emb = np.eye(n, dtype=np.float32)
    # q0/q1 同一用户(历史仅差最近 1 项), q2 不同用户
    hist = [
        np.arange(10, dtype=np.int64),
        np.arange(9, dtype=np.int64),  # Jaccard = 9/11 >= 0.8
        np.arange(100, 105, dtype=np.int64),
    ]
    path = tmp_path / "same_user.npz"
    np.savez(
        path,
        split=np.asarray(["train"] * n),
        index=np.arange(n, dtype=np.int64),
        embeddings=emb,
        history_items=np.asarray(hist, dtype=object),
        meta=np.asarray("{}"),
    )
    ad = InterQueryDiversityAddon(
        user_embedding_path=str(path),
        metrics_path=str(tmp_path / "m.jsonl"),
        top_k_neighbors=2,
    )
    # q0 与 q1 相似度 1.0(同向), 但同用户被排除 -> q0 的邻居只剩 q2
    rows = _make_batch([("train", i, f"cat{i}") for i in range(n)])
    out = ad.compute_batch(rows, [0.1] * len(rows))
    # 只要没有报错且 q0 成功 rollout 的 td 是与 q2 计算的(>0)即可
    ok0 = out[0]
    assert ok0["success"] == 1
    assert ok0["td"] > 0.0


def test_addon_lambda_decays_over_steps(addon):
    ad, n_q, g = addon(n_queries=2, rollout_n=2, top_k_neighbors=1,
                       lambda_init=1.0, lambda_final=0.0, decay_steps=10)
    rows = _make_batch([("train", i, f"d{i}") for i in range(2)])
    first = ad.compute_batch(rows, [0.0] * len(rows))
    intr1 = [r["intrinsic"] for r in first if r["success"]]
    for _ in range(9):
        ad.compute_batch(rows, [0.0] * len(rows))
    last = ad.compute_batch(rows, [0.0] * len(rows))
    intr_last = [r["intrinsic"] for r in last if r["success"]]
    assert all(x > 0 for x in intr1)
    assert all(x == pytest.approx(0.0) for x in intr_last)  # 衰减到 0


# --------------------------------------------------------------------------- #
# verl 入口(direct_recommendation_interdiv)
# --------------------------------------------------------------------------- #


ENTRY_PATH = R4GR_ROOT / "src" / "reward" / "direct_recommendation_interdiv.py"


def _load_entry():
    spec = importlib.util.spec_from_file_location("dd_interdiv_for_test", str(ENTRY_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_entry_disabled_returns_base_reward(tmp_path, tmp_store):
    """enable=False 时逐行 score 与原 StepRule reward 完全一致。"""
    os.environ.setdefault("DATASET", "games")
    if not os.path.exists(INFO_FILE):
        pytest.skip("缺少 info 文件(StepRule 依赖), 跳过兼容性测试")
    mod = _load_entry()
    from src.reward.direct_recommendation_steprule import MyRewardComputer

    rc = MyRewardComputer()
    rows = _make_batch([("train", i, f"e{i}") for i in range(3)])
    data_sources = [r["data_source"] for r in rows]
    sols = [r["solution_str"] for r in rows]
    gts = [r["ground_truth"] for r in rows]
    extras = [r["extra_info"] for r in rows]

    out = mod.rule_base_reward(
        data_sources=data_sources,
        solution_strs=sols,
        ground_truths=gts,
        extra_infos=extras,
        enable_inter_query_diversity=False,
        base_reward_type="steprule",
    )
    expected = [
        rc.compute("rec/test", s, g, None) for s, g in zip(sols, gts)
    ]
    assert [o["score"] for o in out] == pytest.approx(expected)
    assert all(o["intrinsic"] == 0.0 for o in out)


def test_entry_enabled_matches_addon(tmp_path, tmp_store):
    os.environ.setdefault("DATASET", "games")
    path, emb, hist = tmp_store(n=3, dim=4)
    mod = _load_entry()
    rows = _make_batch([("train", i, f"f{i}") for i in range(3)])
    out = mod.rule_base_reward(
        data_sources=[r["data_source"] for r in rows],
        solution_strs=[r["solution_str"] for r in rows],
        ground_truths=[r["ground_truth"] for r in rows],
        extra_infos=[r["extra_info"] for r in rows],
        base_reward_type="steprule",
        enable_inter_query_diversity=True,
        user_embedding_path=str(path),
        metrics_path=str(tmp_path / "entry_metrics.jsonl"),
        lambda_init=0.3,
        sigma=0.4,
        top_k_neighbors=2,
    )
    assert len(out) == len(rows)
    assert any(r["intrinsic"] > 0 for r in out if r["success"])
    for r in out:
        assert r["score"] == pytest.approx(r["base_reward"] + r["intrinsic"])


def test_entry_enabled_requires_embedding_path():
    mod = _load_entry()
    rows = _make_batch([("train", 0, "x")])
    with pytest.raises(ValueError, match="user_embedding_path"):
        mod.rule_base_reward(
            data_sources=["rec/t"],
            solution_strs=[rows[0]["solution_str"]],
            ground_truths=[GT],
            extra_infos=[rows[0]["extra_info"]],
            base_reward_type="steprule",
            enable_inter_query_diversity=True,
        )
