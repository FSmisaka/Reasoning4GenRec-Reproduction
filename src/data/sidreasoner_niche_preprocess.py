"""Niche(小众物品)多样性奖励的离线预处理。

产物(data/raw/Amazon/niche/<Category>/, 均为 reward 端只读字典):

1. popularity.json —— 物品流行度与 niche 标记
   - 频次口径: train split 的 **dedup (user_id, item) 对**(物品的独立
     交互用户数, RecSys 标准 popularity; CSV 是扩展窗口格式, 逐行累加
     会把同一交互重复计入后续行历史);
   - niche = 物品按频次降序排序后, **物品数的后 80%**(频次并列按物品
     id 升序截断, 结果确定可复现); index.json 全量物品参与排名, 零频
     物品自动 niche;
   - 附 sid_to_item(SID 三元组 -> 物品 id, 码本碰撞取最小 id, 与
     sidreasoner_user_embed 同口径), reward 端不再依赖 index.json;
   - 仅依赖 item id, 与官方/自建 SID 集无关; 但 sid_to_item 取自当前
     index.json, **切换 SID 集(switch)后需重跑**。

2. cot_lexstats.json —— CoT 语料(index/<Category>.integrated_narrative.csv
   的 reasoning_path 全量)的词表统计
   - df/counts: 去停用词后的内容词文档频率/词频(停用词 = 仓库
     diversity_analysis 的通用英文表);
   - domain_stopwords: **按语料自动生成的域内停用词**(df >= df_threshold
     * n_docs 的内容词, 如 user/game/prefer), 数据集无关、可迁移;
   - reward 端的 BM25 idf 与 SIF 词频均查本表。

3. sif_words.npz —— SIF 句向量(Arora et al., ICLR 2017)所需静态词向量
   - 词向量 = BGE(词) 的 CLS 输出(单词整句输入, 与 rqvae_embed 同
     模型/口径), 语料词表按 min_count/max_vocab 截断;
   - p(w) = 语料内容词词频占比; common = 语料样本 SIF 句向量的第一
     右奇异向量(公共话语成分, 句向量构建时减去其投影)。

启动:
    DATASET=games make games-sidreasoner-niche-data
    (等价: DATASET=games .venv/bin/python -m src.data.sidreasoner_niche_preprocess)
    可选: --skip-sif(仅生成两个 JSON)/ --force / --device cuda
"""

import argparse
import ast
import datetime
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from config import load_config
from src.analysis.diversity_analysis import STOPWORDS as BASE_STOPWORDS
from src.data.sidreasoner_user_embed import build_sid_to_item
from src.reward.interdiv.niche import tokenize_content
from src.train.common import pick_device, resolve_category


def count_unique_user_items(train_csv: str) -> Counter:
    """train split 的 dedup (user_id, item) 对 -> 物品 -> 独立用户数。

    history_item_id(扩展窗口历史, 含重复行)与 item_id(目标)合并后按
    (user, item) 去重, 消除同一交互在后续行历史中的重复计数。
    """
    df = pd.read_csv(train_csv, usecols=["user_id", "history_item_id", "item_id"])
    pairs = set()
    for uid, hist, tgt in zip(df["user_id"], df["history_item_id"], df["item_id"]):
        pairs.update((uid, int(i)) for i in ast.literal_eval(hist))
        pairs.add((uid, int(tgt)))
    return Counter(item for _, item in pairs)


def build_popularity(index_json: str, pop: Counter, niche_frac: float = 0.8) -> dict:
    """物品流行度/niche 字典(JSON 可序列化)。"""
    with open(index_json, "r", encoding="utf-8") as f:
        index = json.load(f)
    items = sorted(int(k) for k in index)
    sid_to_item = build_sid_to_item(index_json)
    # 频次降序, 并列按物品 id 升序(确定性截断)
    order = sorted(items, key=lambda i: (-pop.get(i, 0), i))
    # round 消除 1-0.8=0.1999... 的浮点截断偏差
    n_popular = int(round(len(items) * (1.0 - niche_frac)))
    popular = set(order[:n_popular])
    boundary = pop.get(order[n_popular - 1], 0) if order and n_popular > 0 else 0
    return {
        "meta": {
            "n_items": len(items),
            "niche_frac": float(niche_frac),
            "counting": "unique_user_pairs",
            "n_popular": n_popular,
            "n_niche": len(items) - n_popular,
            "popularity_boundary": int(boundary),
            "index_file": os.path.abspath(index_json),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "sid_to_item": sid_to_item,
        "items": {
            str(i): {"pop": int(pop.get(i, 0)), "niche": i not in popular}
            for i in items
        },
    }


def iter_cot_corpus(narrative_csv: str):
    df = pd.read_csv(narrative_csv, usecols=["reasoning_path"])
    for s in df["reasoning_path"]:
        yield str(s)


def build_lexstats(docs, df_threshold: float = 0.5) -> dict:
    """CoT 语料词表统计(df/counts/域内停用词, 均基于内容词)。"""
    df_counter, tf_counter = Counter(), Counter()
    n_docs, total_tokens, doc_lens = 0, 0, []
    for doc in docs:
        tokens = tokenize_content(doc, BASE_STOPWORDS)
        if not tokens:
            continue
        n_docs += 1
        total_tokens += len(tokens)
        doc_lens.append(len(tokens))
        tf_counter.update(tokens)
        df_counter.update(set(tokens))
    domain_stopwords = sorted(
        w for w, d in df_counter.items() if d >= df_threshold * max(n_docs, 1)
    )
    return {
        "meta": {
            "n_docs": n_docs,
            "df_threshold": float(df_threshold),
            "n_domain_stopwords": len(domain_stopwords),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "n_docs": n_docs,
        "total_tokens": total_tokens,
        "avg_doc_len": float(np.mean(doc_lens)) if doc_lens else 0.0,
        "df": dict(df_counter),
        "counts": dict(tf_counter),
        "domain_stopwords": domain_stopwords,
    }


def build_sif(vocab: list, counts: dict, embeddings: np.ndarray,
              token_lists: list, a: float = 1e-3) -> dict:
    """SIF 静态资产: 词频 p(w) + 语料公共成分(第一右奇异向量)。

    Args:
        vocab: 词表(与 embeddings 行对齐);
        counts: 语料内容词词频(p(w) 的分子);
        embeddings: (V, dim) 已 L2 归一化的静态词向量;
        token_lists: 语料(采样后的)分词文档, 用于估计公共成分。
    """
    index = {w: i for i, w in enumerate(vocab)}
    total = max(sum(counts.get(w, 0) for w in vocab), 1)
    p = np.asarray(
        [counts.get(w, 0) / total for w in vocab], dtype=np.float64
    )
    rows = []
    for toks in token_lists:
        idxs = [index[t] for t in toks if t in index]
        if not idxs:
            continue
        weights = a / (a + p[idxs])
        rows.append(
            (embeddings[idxs].astype(np.float64) * weights[:, None]).sum(axis=0)
            / len(idxs)
        )
    common = np.zeros(embeddings.shape[1], dtype=np.float64)
    if len(rows) >= 2:
        mat = np.asarray(rows)
        _, _, vt = np.linalg.svd(mat, full_matrices=False)
        common = vt[0]
    return {"p": p, "common": common}


def encode_vocab_words(vocab: list, model: str, device: str, batch_size: int):
    """词表 -> BGE 静态词向量(单词整句输入, CLS pooling + L2 归一化)。"""
    from src.data.rqvae_embed import encode_corpus

    return encode_corpus(list(vocab), model, batch_size, 512, device)


def niche_out_dir(root: str, category: str) -> str:
    return os.path.join(root, "niche", category)


def main(argv=None, cfg=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None,
                        help="产物目录(默认 <root>/niche/<Category>)")
    parser.add_argument("--niche-frac", type=float, default=0.8,
                        help="niche 物品占比(按物品数, 默认后 80%%)")
    parser.add_argument("--df-threshold", type=float, default=0.5,
                        help="域内停用词的文档频率阈值(占比, 默认 0.5)")
    parser.add_argument("--skip-sif", action="store_true",
                        help="跳过 sif_words.npz(仅用 tfidf 变体时)")
    parser.add_argument("--sif-min-count", type=int, default=5)
    parser.add_argument("--sif-max-vocab", type=int, default=100_000)
    parser.add_argument("--sif-sample", type=int, default=20_000,
                        help="估计 SIF 公共成分的语料采样文档数")
    parser.add_argument("--sif-a", type=float, default=1e-3)
    parser.add_argument("--model", default=None,
                        help="词向量模型(默认 config rqvae embed.model)")
    parser.add_argument("--device", default=None, help="cuda|mps|cpu(默认自动)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="已存在也重算")
    args = parser.parse_args(argv)

    cfg = cfg or load_config("sidreasoner")
    category = resolve_category(cfg["data"])
    root = cfg["data"]["root"]
    out_dir = args.out_dir or niche_out_dir(root, category)
    os.makedirs(out_dir, exist_ok=True)

    train_csv = os.path.join(root, "train", f"{category}_5_2016-10-2018-11.csv")
    narrative_csv = os.path.join(root, "index", f"{category}.integrated_narrative.csv")
    index_json = os.path.join(root, "index", f"{category}.index.json")
    for path, what in (
        (train_csv, "train split CSV"),
        (narrative_csv, "CoT 语料(integrated_narrative.csv)"),
        (index_json, "index.json"),
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到{what}: {path}")

    # ---- 1) popularity.json ----
    pop_path = os.path.join(out_dir, "popularity.json")
    if args.force or not os.path.exists(pop_path):
        pop_counter = count_unique_user_items(train_csv)
        payload = build_popularity(index_json, pop_counter, args.niche_frac)
        with open(pop_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        m = payload["meta"]
        print(
            f"[niche-prep] popularity: {m['n_items']} items "
            f"(popular {m['n_popular']} / niche {m['n_niche']}, "
            f"边界频次 {m['popularity_boundary']}) -> {pop_path}"
        )
    else:
        print(f"[niche-prep] 已存在, 跳过(--force 重算): {pop_path}")

    # ---- 2) cot_lexstats.json ----
    lex_path = os.path.join(out_dir, "cot_lexstats.json")
    if args.force or not os.path.exists(lex_path):
        lexstats = build_lexstats(iter_cot_corpus(narrative_csv), args.df_threshold)
        with open(lex_path, "w", encoding="utf-8") as f:
            json.dump(lexstats, f)
        print(
            f"[niche-prep] lexstats: {lexstats['n_docs']} docs, "
            f"vocab {len(lexstats['df'])}, 域内停用词 "
            f"{lexstats['meta']['n_domain_stopwords']} -> {lex_path}"
        )
    else:
        print(f"[niche-prep] 已存在, 跳过(--force 重算): {lex_path}")

    # ---- 3) sif_words.npz ----
    sif_path = os.path.join(out_dir, "sif_words.npz")
    if args.skip_sif:
        print("[niche-prep] --skip-sif, 跳过 sif_words.npz")
        return
    if not args.force and os.path.exists(sif_path):
        print(f"[niche-prep] 已存在, 跳过(--force 重算): {sif_path}")
        return
    with open(lex_path, "r", encoding="utf-8") as f:
        lex = json.load(f)
    counts = lex["counts"]
    vocab = [
        w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if c >= args.sif_min_count
    ][: args.sif_max_vocab]
    if not vocab:
        raise ValueError("语料词表为空, 无法构建 SIF 词向量。")
    model = args.model or load_config("rqvae")["embed"]["model"]
    device = args.device or pick_device()
    print(f"[niche-prep] SIF: {len(vocab)} 词 x BGE({model}) on {device} ...")
    embeddings = encode_vocab_words(vocab, model, device, args.batch_size)

    docs = list(iter_cot_corpus(narrative_csv))
    rng = np.random.default_rng(42)
    if len(docs) > args.sif_sample:
        docs = [docs[i] for i in rng.choice(len(docs), args.sif_sample, replace=False)]
    token_lists = [tokenize_content(d, BASE_STOPWORDS) for d in docs]
    sif = build_sif(vocab, counts, embeddings, token_lists, a=args.sif_a)
    meta = {
        "model": model,
        "a": float(args.sif_a),
        "min_count": args.sif_min_count,
        "max_vocab": args.sif_max_vocab,
        "n_vocab": len(vocab),
        "common_sample": len(token_lists),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    np.savez(
        sif_path,
        vocab=np.asarray(vocab, dtype=str),
        embeddings=embeddings.astype(np.float32),
        p=sif["p"],
        common=sif["common"],
        meta=np.asarray(json.dumps(meta)),
    )
    print(f"[niche-prep] SIF 资产 -> {sif_path} ({meta})")
    return out_dir


if __name__ == "__main__":
    main()
