"""Niche(小众物品)多样性奖励单测。

覆盖: 离线预处理(popularity/lexstats/SIF)、scorer 三分量(c1/c2/c3)、
总结段抽取边界、compute_many 聚合, 以及 InterQueryDiversityAddon 的
similarity="niche" 分支(注入 stub 句向量编码器, 不加载 BGE)。

运行: .venv/bin/python -m pytest tests/test_niche.py -v
"""

import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

R4GR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R4GR_ROOT))

from src.data.sidreasoner_niche_preprocess import (  # noqa: E402
    build_lexstats,
    build_popularity,
    build_sif,
)
from src.reward.interdiv.niche import (  # noqa: E402
    NicheDiversityScorer,
    tokenize_content,
)
from src.reward.interdiv.reward import InterQueryDiversityAddon  # noqa: E402

GT = "<a_1><b_1><c_1>"


def _resp(cot, answer=GT):
    return f"<think>\n{cot}\n</think>\n\n{answer}"


# --------------------------------------------------------------------------- #
# 离线预处理
# --------------------------------------------------------------------------- #

POP = {0: 9, 1: 5, 2: 3, 3: 2, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0}


def _write_index(tmp_path, n_items=10):
    index = {
        str(i): [f"<a_{i}>", f"<b_{i}>", f"<c_{i}>"] for i in range(n_items)
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    return str(path)


def test_build_popularity_rank_cut_and_ties(tmp_path):
    index_json = _write_index(tmp_path)
    pop = build_popularity(index_json, Counter(POP), niche_frac=0.8)
    # 10 物品 -> top 2 popular(9, 5 用户), 其余 8 个 niche(含零频的 8/9)
    assert pop["meta"]["n_popular"] == 2 and pop["meta"]["n_niche"] == 8
    assert pop["items"]["0"]["niche"] is False
    assert pop["items"]["1"]["niche"] is False
    assert all(pop["items"][str(i)]["niche"] for i in range(2, 10))
    assert pop["items"]["8"]["pop"] == 0  # 零频物品保留且 niche
    # sid_to_item: 三元组拼接串 -> 最小物品 id
    assert pop["sid_to_item"]["<a_2><b_2><c_2>"] == 2
    # 频次并列(4..7 都是 1)时按物品 id 升序截断 -> 结果确定
    pop2 = build_popularity(index_json, Counter(POP), niche_frac=0.8)
    assert pop2["items"] == pop["items"]


def test_build_lexstats_df_and_domain_stopwords():
    docs = [
        "The user enjoys <a_0><b_0><c_0> adventure titles and racing wheels",
        "The user prefers <a_1><b_1><c_1> puzzle quests about treasure maps",
        "The user likes adventure stories with mystery and suspense arcs",
        "A quiet story about folklore legends of the northern islands",
    ]
    lex = build_lexstats(docs, df_threshold=0.5)
    assert lex["n_docs"] == 4
    # SID token 与基础英文停用词不进词表
    assert "<a_0>" not in lex["df"] and "the" not in lex["df"]
    # "user" 出现在 3/4 文档 -> 域内停用词; "adventure" 2/4 -> 同样超过 0.5
    assert "user" in lex["domain_stopwords"]
    assert "adventure" in lex["domain_stopwords"]
    # "folklore" 只在 1 篇 -> 内容词
    assert "folklore" not in lex["domain_stopwords"]
    assert lex["df"]["folklore"] == 1
    assert lex["counts"]["racing"] == 1


def test_build_sif_probs_and_common():
    vocab = ["adventure", "racing", "puzzle", "folklore"]
    counts = {"adventure": 6, "racing": 3, "puzzle": 1, "folklore": 0}
    emb = np.eye(4, dtype=np.float32)
    # 公共成分估计语料用词表外词 -> 无有效行 -> common 置零
    sif = build_sif(vocab, counts, emb, [["zzz"], ["yyy"]])
    assert np.allclose(sif["common"], 0.0)
    p = sif["p"]
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2] > p[3]
    # 有有效行时 common 为单位向量(SVD 首右奇异向量)
    sif2 = build_sif(vocab, counts, emb, [["adventure", "racing"], ["puzzle", "adventure"]])
    assert np.linalg.norm(sif2["common"]) == pytest.approx(1.0)


def test_tokenize_content_strips_sid_and_stopwords():
    toks = tokenize_content(
        "The USER enjoys <a_0><b_0><c_0> odd's adventure titles", {"the", "user"}
    )
    assert toks == ["enjoys", "odd's", "adventure", "titles"]


# --------------------------------------------------------------------------- #
# 测试资产(scorer)
# --------------------------------------------------------------------------- #

SIF_VOCAB = ["adventure", "racing", "puzzle", "folklore"]


class DictEncoder:
    """stub 句子编码器: 指定文本 -> 指定向量, 其余 crc32 哈希 one-hot。"""

    def __init__(self, dim=8):
        self.dim = dim
        self.vec_map = {}

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            v = self.vec_map.get(t)
            if v is None:
                v = np.zeros(self.dim, dtype=np.float64)
                v[zlib.crc32(t.encode()) % self.dim] = 1.0
            out[i] = v
        return out


@pytest.fixture
def scorer_assets(tmp_path):
    """10 物品(0/1 popular, 2..9 niche)+ 4 词 SIF + 小语料 lexstats。"""
    index_json = _write_index(tmp_path)
    pop = build_popularity(index_json, Counter(POP), niche_frac=0.8)
    pop_path = tmp_path / "popularity.json"
    pop_path.write_text(json.dumps(pop))

    corpus = [
        "The user enjoys <a_0><b_0><c_0> adventure titles and racing wheels",
        "The user prefers <a_1><b_1><c_1> puzzle quests about treasure maps",
        "The user likes adventure stories with mystery and suspense arcs",
        "A quiet folklore story about legends of the northern islands",
    ]
    lex = build_lexstats(corpus, df_threshold=0.9)  # 高阈值 -> 无域内停用词
    lex_path = tmp_path / "cot_lexstats.json"
    lex_path.write_text(json.dumps(lex))

    emb = np.eye(len(SIF_VOCAB), dtype=np.float32)
    sif = build_sif(SIF_VOCAB, {w: 1 for w in SIF_VOCAB}, emb, [])
    sif_path = tmp_path / "sif_words.npz"
    np.savez(
        sif_path,
        vocab=np.asarray(SIF_VOCAB, dtype=str),
        embeddings=emb,
        p=sif["p"],
        common=sif["common"],
    )
    return {
        "pop": str(pop_path),
        "lex": str(lex_path),
        "sif": str(sif_path),
        "tmp": tmp_path,
    }


def _make_scorer(assets, method="sif", encoder=None):
    # 默认注入 stub 编码器(c2 的句子 embedding), 避免 lazy 加载 BGE
    return NicheDiversityScorer(
        niche_dict_path=assets["pop"],
        lexstats_path=assets["lex"],
        summary_method=method,
        sif_path=assets["sif"] if method == "sif" else None,
        encoder=encoder if encoder is not None else DictEncoder(),
    )


# --------------------------------------------------------------------------- #
# c1: 信息提取比例
# --------------------------------------------------------------------------- #

def test_c1_ratio_set_semantics_and_history_restriction(scorer_assets):
    s = _make_scorer(scorer_assets)
    cot = (
        "The user played <a_0><b_0><c_0> and loved it. "
        "The niche title <a_2><b_2><c_2> looks obscure, as does <a_3><b_3><c_3>. "
        "Consider these niche picks next."
    )
    # 历史 [0(popular), 2, 3] -> niche_hist = {2,3}; CoT 提到 2/3 各一次
    d, diag = s.compute_one(cot, refs=[], history_items=[0, 2, 3])
    assert diag["niche_c1"] == pytest.approx(1.0)
    assert diag["niche_hist_count"] == 2 and diag["niche_mentioned_count"] == 2

    # 历史中 niche 物品重复出现 -> 分母去重只计一次; CoT 只提到 2
    cot2 = "About <a_2><b_2><c_2> only. And <a_2><b_2><c_2> again. Summarize."
    d2, diag2 = s.compute_one(cot2, refs=[], history_items=[2, 2, 3])
    assert diag2["niche_c1"] == pytest.approx(0.5)

    # CoT 提到历史外的 niche 物品(5)与幻觉 SID(77) -> 不进分子
    cot3 = "Rare pick <a_5><b_5><c_5> and fake <a_77><b_77><c_77>. Done."
    d3, diag3 = s.compute_one(cot3, refs=[], history_items=[2, 3])
    assert diag3["niche_c1"] == pytest.approx(0.0)

    # 历史无 niche 物品 -> 分母 0 -> c1 = 0(未定义)
    d4, diag4 = s.compute_one("Any cot.", refs=[], history_items=[0, 1])
    assert diag4["niche_c1"] == 0.0 and diag4["niche_c1_defined"] == 0


# --------------------------------------------------------------------------- #
# c2: 小众大众句相似度
# --------------------------------------------------------------------------- #

POP_SENT = "The user played <a_0><b_0><c_0> and loved it."
NICHE_SENT = "The niche title <a_2><b_2><c_2> looks obscure."
MIXED_SENT = "The user played <a_0><b_0><c_0> and the obscure <a_2><b_2><c_2> together."
TAIL = "Consider all of these observations for the final choice."


def test_c2_mixed_sentences_dropped_from_both(scorer_assets):
    s = _make_scorer(scorer_assets)
    niche_s, pop_s = s.classify_sentences(f"{POP_SENT} {NICHE_SENT} {MIXED_SENT} {TAIL}")
    assert niche_s == {" ".join(NICHE_SENT.split())}
    assert pop_s == {" ".join(POP_SENT.split())}
    # mixed 句不出现在任何一侧
    mixed_key = " ".join(MIXED_SENT.split())
    assert mixed_key not in niche_s and mixed_key not in pop_s


def test_c2_orthogonal_and_identical_sentences(scorer_assets):
    enc = DictEncoder(dim=4)
    e_pop = np.array([1.0, 0.0, 0.0, 0.0])
    e_niche = np.array([0.0, 1.0, 0.0, 0.0])
    enc.vec_map[" ".join(POP_SENT.split())] = e_pop
    enc.vec_map[" ".join(NICHE_SENT.split())] = e_niche
    cot = f"{POP_SENT} {NICHE_SENT} {TAIL}"

    s = _make_scorer(scorer_assets, encoder=enc)
    _, diag = s.compute_one(cot, refs=[], history_items=[0, 2])
    assert diag["niche_c2_defined"] == 1
    assert diag["niche_c2"] == pytest.approx(1.0)  # 正交 -> sim 0

    # 同一方向的向量 -> sim 1 -> c2 = 0
    enc2 = DictEncoder(dim=4)
    enc2.vec_map[" ".join(POP_SENT.split())] = e_pop
    enc2.vec_map[" ".join(NICHE_SENT.split())] = e_pop.copy()
    s2 = _make_scorer(scorer_assets, encoder=enc2)
    _, diag2 = s2.compute_one(cot, refs=[], history_items=[0, 2])
    assert diag2["niche_c2"] == pytest.approx(0.0)


def test_c2_zero_when_either_side_empty(scorer_assets):
    s = _make_scorer(scorer_assets)
    # 只有 niche 句(历史与 CoT 全 niche)
    cot = f"{NICHE_SENT} {TAIL}"
    _, diag = s.compute_one(cot, refs=[], history_items=[2, 3])
    assert diag["niche_c2"] == 0.0 and diag["niche_c2_defined"] == 0
    # 只有 popular 句
    cot2 = f"{POP_SENT} {TAIL}"
    _, diag2 = s.compute_one(cot2, refs=[], history_items=[0, 2])
    assert diag2["niche_c2"] == 0.0 and diag2["niche_c2_defined"] == 0
    # 完全没有含 SID 的句子
    _, diag3 = s.compute_one("plain text without any sids here.", refs=[], history_items=[2])
    assert diag3["niche_c2"] == 0.0 and diag3["niche_c2_defined"] == 0


# --------------------------------------------------------------------------- #
# 总结段抽取与 c3
# --------------------------------------------------------------------------- #

def test_extract_summary_boundaries():
    cot = f"{POP_SENT} {NICHE_SENT} {TAIL}"
    assert NicheDiversityScorer.extract_summary(cot) == TAIL
    # 最后一个 SID 所在句之后没有内容 -> 空
    assert NicheDiversityScorer.extract_summary(f"{POP_SENT} {NICHE_SENT}") == ""
    # 无 SID -> 空
    assert NicheDiversityScorer.extract_summary("no sids at all. none.") == ""
    # 句界也可以是换行
    cot_nl = f"{POP_SENT}\n{NICHE_SENT}\n{TAIL}"
    assert NicheDiversityScorer.extract_summary(cot_nl) == TAIL


def test_c3_tfidf_identical_disjoint_and_unseen_max_idf(scorer_assets):
    enc = DictEncoder()
    own_cot = f"{NICHE_SENT} adventure racing wheels dominate."
    same_ref = f"{POP_SENT} adventure racing wheels dominate."
    diff_ref = f"{POP_SENT} puzzle folklore quests fill evenings."
    s = _make_scorer(scorer_assets, method="tfidf", encoder=enc)

    # 相同总结段(内容词一致) -> sim=1 -> c3=0
    _, d_same = s.compute_one(own_cot, refs=[same_ref], history_items=[2])
    assert d_same["niche_c3"] == pytest.approx(0.0)
    assert d_same["niche_c3_defined"] == 1

    # 无共享内容词 -> 余弦 0 -> c3 = 1
    _, d_diff = s.compute_one(own_cot, refs=[diff_ref], history_items=[2])
    assert d_diff["niche_c3"] == pytest.approx(1.0)

    # 未登录词取最大 idf: 出现未登录词的总结段与只含高频词的总结段相比,
    # 未登录词权重 == max_idf(与 df=0 的词同权)
    vec_zzz = s._tfidf_vector(["zzzunseen"])
    assert vec_zzz["zzzunseen"] == pytest.approx(s.max_idf)

    # 无有效 reference -> c3 = 0(未定义), c1/c2 照常计算
    d_none, diag = s.compute_one(own_cot, refs=[], history_items=[2, 3])
    assert diag["niche_c3"] == 0.0 and diag["niche_c3_defined"] == 0
    assert diag["niche_c1_defined"] == 1

    # 总结段为空(最后一个 SID 就在末句) -> c3 = 0
    _, diag2 = s.compute_one(f"{NICHE_SENT} {POP_SENT}", refs=[same_ref], history_items=[2])
    assert diag2["niche_c3"] == 0.0 and diag2["niche_c3_defined"] == 0


def test_c3_sif_identical_and_disjoint(scorer_assets):
    enc = DictEncoder()
    # 4 词正交词向量; common 置零 -> 句向量 = 加权平均后归一
    own_cot = f"{NICHE_SENT} adventure racing appeal."
    same_ref = f"{POP_SENT} racing adventure appeal."
    diff_ref = f"{POP_SENT} puzzle folklore appeal."
    s = _make_scorer(scorer_assets, method="sif", encoder=enc)

    _, d_same = s.compute_one(own_cot, refs=[same_ref], history_items=[2])
    assert d_same["niche_c3"] == pytest.approx(0.0, abs=1e-9)

    _, d_diff = s.compute_one(own_cot, refs=[diff_ref], history_items=[2])
    assert d_diff["niche_c3"] == pytest.approx(1.0)

    # 词表外词被跳过; 全部词表外 -> 向量 None -> c3 = 0
    _, d_oov = s.compute_one(
        f"{NICHE_SENT} qwerty uiop asdf.", refs=[same_ref], history_items=[2]
    )
    assert d_oov["niche_c3"] == 0.0 and d_oov["niche_c3_defined"] == 0


def test_compute_many_averages_three_components(scorer_assets):
    enc = DictEncoder(dim=4)
    enc.vec_map[" ".join(POP_SENT.split())] = np.array([1.0, 0, 0, 0])
    enc.vec_map[" ".join(NICHE_SENT.split())] = np.array([0, 1.0, 0, 0])
    s = _make_scorer(scorer_assets, encoder=enc)

    cot = f"{NICHE_SENT} {POP_SENT} adventure racing appeal."
    ref = f"{POP_SENT} puzzle folklore appeal."
    items = [
        # c1=1(niche 2 全提), c2=1(正交句), c3=1(SIF 词表正交, appeal 词表外)
        {"cot": cot, "refs": [ref], "history_items": [0, 2]},
        # c1=0(历史无 niche), c2=0(无 niche 句), c3=0(无 ref)
        {"cot": f"{POP_SENT} {TAIL}", "refs": [], "history_items": [0, 1]},
    ]
    out = s.compute_many(items)
    assert out[0][0] == pytest.approx(1.0)
    assert out[1][0] == pytest.approx(0.0)
    # batch 级统计
    assert s.last_stats["n_scored"] == 2
    assert s.last_stats["c1_mean"] == pytest.approx(0.5)
    assert s.last_stats["c2_defined_frac"] == pytest.approx(0.5)
    # history_items 允许 np.ndarray
    out_np = s.compute_many(
        [{"cot": cot, "refs": [ref], "history_items": np.asarray([0, 2])}]
    )
    assert out_np[0][0] == pytest.approx(1.0)


def test_scorer_missing_artifacts_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="niche-data"):
        NicheDiversityScorer(
            niche_dict_path=str(tmp_path / "nope.json"),
            lexstats_path=str(tmp_path / "nope2.json"),
        )
    with pytest.raises(ValueError):
        NicheDiversityScorer(
            niche_dict_path="x", lexstats_path="y", summary_method="bogus"
        )


# --------------------------------------------------------------------------- #
# Addon 集成(similarity="niche")
# --------------------------------------------------------------------------- #

@pytest.fixture
def niche_addon(tmp_path, scorer_assets):
    def make(n_queries=3, rollout_n=2, **kw):
        n, dim = n_queries, 8
        emb = np.eye(dim, dtype=np.float32)[:n]
        if n >= 2:
            emb[1] = emb[0]  # q1 与 q0 同向 -> q0 的 top1 邻居确定是 q1
        # 历史: q0 [0,2] / q1 [0,1] / q2 [2,9]; niche = {2..9}
        hists = [[0, 2], [0, 1], [2, 9]][:n]
        path = tmp_path / f"user_emb_{n_queries}.npz"
        np.savez(
            path,
            split=np.asarray(["train"] * n),
            index=np.arange(n, dtype=np.int64),
            embeddings=emb,
            history_items=np.asarray(
                [np.asarray(h, dtype=np.int64) for h in hists], dtype=object
            ),
            meta=np.asarray("{}"),
        )
        return (
            InterQueryDiversityAddon(
                user_embedding_path=str(path),
                metrics_path=str(tmp_path / "metrics.jsonl"),
                top_k_neighbors=1,
                similarity="niche",
                success_mode="none",
                reference_mode="none",
                lambda_init=1.0,
                sigma=1.0,
                niche_dict_path=scorer_assets["pop"],
                niche_lexstats_path=scorer_assets["lex"],
                niche_sif_path=scorer_assets["sif"],
                niche_encoder=DictEncoder(dim=4),
                **kw,
            ),
            n_queries,
            rollout_n,
        )

    return make


def _niche_batch(queries, rollout_n=2):
    rows = []
    for qi, (split, idx, cot) in enumerate(queries):
        for g in range(rollout_n):
            rows.append(
                {
                    "data_source": "rec/test",
                    "solution_str": _resp(cot),
                    "ground_truth": GT,
                    "extra_info": {"split": split, "index": idx},
                }
            )
    return rows


def test_addon_niche_td_and_diag(niche_addon):
    ad, n_q, g = niche_addon()
    own = f"{NICHE_SENT} {POP_SENT} adventure racing appeal."
    nbr = f"{NICHE_SENT} {POP_SENT} puzzle folklore appeal."
    rows = _niche_batch([("train", 0, own), ("train", 1, nbr), ("train", 2, own)])
    out = ad.compute_batch(rows, [0.5] * len(rows))
    for r in out:
        assert "niche_c1" in r and "niche_c3" in r
        assert r["td"] == pytest.approx(
            (r["niche_c1"] + r["niche_c2"] + r["niche_c3"]) / 3.0
        )
        # success_mode=none -> 可解析即塑形; td<=1 被 sigma=1 clip 不变
        assert 0.0 <= r["td"] <= 1.0
    # q0: c1 = 1(niche 2 在历史), c2 = 1(正交句), c3 = 1(词表正交, ref=q1)
    q0 = out[0]
    assert q0["niche_c1"] == pytest.approx(1.0)
    assert q0["niche_c2"] == pytest.approx(1.0)
    assert q0["niche_c3"] == pytest.approx(1.0)
    assert q0["td"] == pytest.approx(1.0)
    assert q0["intrinsic"] == pytest.approx(1.0)  # lambda=1, sigma=1, I=1
    assert q0["score"] == pytest.approx(1.5)
    # q1: 历史 [0,1] 无 niche -> c1 未定义; ref 来自 q0(同向)
    q1_rows = out[g : 2 * g]
    assert all(r["niche_c1_defined"] == 0 for r in q1_rows)
    # metrics 含 niche 统计
    rec = json.loads(open(ad.metrics_path).read().strip().splitlines()[0])
    assert "niche/c1_mean" in rec and "niche/c2_defined_frac" in rec
    assert rec["similarity"] == "niche"


def test_addon_niche_c1_c2_without_refs(niche_addon):
    """K_q=0(无邻居 reference)时 c3=0, 但 c1/c2 照常计算(设计如此)。"""
    ad, n_q, g = niche_addon(n_queries=2)
    own = f"{NICHE_SENT} {POP_SENT} adventure racing appeal."
    rows = _niche_batch([("train", 0, own), ("train", 1, "placeholder")])
    # q1 的 rollout 全部不可解析(无 </think>) -> q0 无 reference
    broken = "<think> never closed " + "word " * 30
    for r in rows[g:]:
        r["solution_str"] = broken
    out = ad.compute_batch(rows, [0.2] * len(rows))
    ok_rows = out[:g]
    assert all(r["n_refs"] == 0 for r in ok_rows)
    assert all(r["niche_c1"] == pytest.approx(1.0) for r in ok_rows)
    assert all(r["niche_c2_defined"] == 1 for r in ok_rows)
    assert all(r["niche_c3"] == 0.0 for r in ok_rows)
    assert all(
        r["td"] == pytest.approx((1.0 + r["niche_c2"]) / 3.0) for r in ok_rows
    )


def test_addon_niche_diag_keys_consistent(niche_addon):
    """未评分 rollout(CoT 截断/过短)也必须带全部 niche 诊断键。

    回归: verl BatchRewardManager 按 key 聚合逐行 dict 为等长数组;
    缺键会导致 batch 内长度不一致(update_actor -> check_consistency
    断言 "key niche_c1 length N != batch size M")。
    """
    ad, n_q, g = niche_addon(n_queries=2)
    own = f"{NICHE_SENT} {POP_SENT} adventure racing appeal."
    rows = _niche_batch([("train", 0, own), ("train", 1, "placeholder")])
    # q0 第 1 条: CoT 未闭合(extract_cot -> None)
    rows[1]["solution_str"] = "broken without close tag " + "word " * 10
    # q1 第 1 条: CoT 过短(< 5 token)
    rows[g + 1]["solution_str"] = _resp("tiny")
    out = ad.compute_batch(rows, [0.0] * len(rows))
    # 所有行的键集必须一致
    keysets = {frozenset(r.keys()) for r in out}
    assert len(keysets) == 1, keysets
    # 未评分行取默认 0(而不是缺键)
    for r in (out[1], out[g + 1]):
        assert r["niche_c1"] == 0.0 and r["niche_c1_defined"] == 0
        assert r["niche_c2"] == 0.0 and r["niche_c3"] == 0.0
    # 评分行仍是真实分解值
    assert out[0]["niche_c1"] == pytest.approx(1.0)


def test_addon_niche_only_success_skips_failures(niche_addon):
    """默认只对成功 rollout 计算 c1/c2/c3(省 BGE 计算); 关掉后失败行也计算。"""
    ad, n_q, g = niche_addon(n_queries=2)
    ad.success_mode = "prefix"  # GT 首层匹配才算成功
    own = f"{NICHE_SENT} {POP_SENT} adventure racing appeal."
    rows = _niche_batch([("train", 0, own), ("train", 1, own)])
    # 每条 query 第 0 条成功, 第 1 条首层不匹配 -> 失败
    for qi in range(n_q):
        rows[qi * g + 1]["solution_str"] = _resp(own, "<a_9><b_9><c_9>")
    out = ad.compute_batch(rows, [0.0] * len(rows))
    # q0(历史 [0,2]) 成功行正常评分; 失败行被跳过(td=0 且诊断取默认)
    assert out[0]["success"] == 1 and out[0]["niche_c1"] == pytest.approx(1.0)
    assert out[1]["success"] == 0
    assert out[1]["td"] == 0.0 and out[1]["niche_c1"] == 0.0
    # q1(历史 [0,1], 无 niche) 成功行 c1 未定义, 但仍被评分(c2)
    assert out[2]["success"] == 1 and out[2]["niche_c1_defined"] == 0
    assert out[3]["td"] == 0.0
    # 关闭 niche_only_success: 失败行也计算 d(但 intrinsic 仍为 0)
    ad2, _, _ = niche_addon(n_queries=2, niche_only_success=False)
    ad2.success_mode = "prefix"
    out2 = ad2.compute_batch(rows, [0.0] * len(rows))
    assert out2[1]["success"] == 0
    assert out2[1]["niche_c1"] == pytest.approx(1.0)
    assert out2[1]["td"] > 0.0  # c1=1 -> d>=1/3
    assert out2[1]["intrinsic"] == 0.0


def test_addon_niche_rejects_missing_paths(tmp_path):
    """niche 产物缺失时构造报错(指向 make <ds>-sidreasoner-niche-data)。"""
    n, dim = 2, 4
    path = tmp_path / "emb.npz"
    np.savez(
        path,
        split=np.asarray(["train"] * n),
        index=np.arange(n, dtype=np.int64),
        embeddings=np.eye(dim, dtype=np.float32)[:n],
        history_items=np.asarray([np.arange(2)] * n, dtype=object),
        meta=np.asarray("{}"),
    )
    with pytest.raises(FileNotFoundError, match="niche-data"):
        InterQueryDiversityAddon(
            user_embedding_path=str(path),
            metrics_path=str(tmp_path / "m.jsonl"),
            similarity="niche",
            niche_dict_path=str(tmp_path / "nope_pop.json"),
            niche_lexstats_path=str(tmp_path / "nope_lex.json"),
            niche_sif_path=str(tmp_path / "nope_sif.npz"),
        )
