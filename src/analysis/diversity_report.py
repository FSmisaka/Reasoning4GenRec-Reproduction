"""
思维链多样性分析 · 阶段二: 统计与报告

读取 collect_diversity_corpora.py 产出的语料(results/diversity/<Category>/
corpora/*.jsonl), 对每个语料从两个视角统计:

  - text 视角: 剔除 SID token(<a_N><b_N><c_N>)后只看自然语言,
    检验"推理话术是否套路化";
  - raw  视角: 保留 SID token, 反映文档整体(含引用了哪些历史物品)
    的重复程度。

统计项(每个语料 × 两个视角; 词汇类与 gzip/Jaccard 另有去停用词对照版):
  1. 词频分析: TF-IDF(每条思维链一个文档, sublinear tf + smooth idf
     + L2 归一化), 报告平均权重 Top 词; 另给去停用词后的原始词频 Top;
  2. 词汇二元组(bigram)频率 Top;
  3. 词汇多样性: 词表大小 / TTR / 语料级与文档级 distinct-1,2 /
     unigram, bigram 熵(归一化)——以上均有 含停用词(lexical, 与历史
     口径一致) 与 去停用词(lexical_content, 仅内容词) 两个版本;
  4. 重复度: 规范化文本的完全重复率; 成对 TF-IDF 余弦相似度
     (mean/median/p95/max, >0.6, >0.8 占比); 成对词 3-gram Jaccard
     (duplication.jaccard3 含停用词 / jaccard3_content 去停用词);
  5. 首句模板集中度: 首句(SID/数字泛化后)的 Top 覆盖率与唯一率;
  6. gzip 压缩率(gzip_ratio 原文 / gzip_ratio_content 去停用词拼接;
     越低越多样);
  7. 最相似样本对示例(按余弦 Top-3, 便于人工核查"套路"长什么样)。

文档数超过 --max-docs 时按 seed 随机采样后再统计; 成对指标再受
--pair-cap 限制(计算量 O(n^2))。输出:
  results/diversity/<Category>/analysis.json  全部数值
  results/diversity/<Category>/report.md      人读对比报告

用法:
  DATASET=games .venv/bin/python -m src.analysis.diversity_report
  可选: --corpora teacher,stage2 --top 25 --pair-cap 600 --max-docs 2000
  全量: --full(= --max-docs 0, 不采样文档; 成对指标仍受 --pair-cap 限制,
        建议按机器上调 PAIR_CAP, 如 2000)
"""

import argparse
import gzip
import json
import math
import os
import random
import re
import sys
from collections import Counter

import numpy as np

SID_RE = re.compile(r"<[abc]_\d+>")
WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+")

STOPWORDS = set(
    """a about above after again against all am an and any are aren't as at
    be because been before being below between both but by can cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he her here hers
    herself him himself his how i if in into is isn't it its itself let's me
    more most mustn't my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan't she should shouldn't so some
    such than that the their theirs them themselves then there these they
    this those through to too under until up very was wasn't we were weren't
    what when where which while who whom why with won't would wouldn't you
    your yours yourself yourselves""".split()
)


# --------------------------------------------------------------------------
# 分词
# --------------------------------------------------------------------------
def tokenize(text, keep_sid):
    """返回 token 列表。keep_sid=True 时 SID token 作为一个词保留;
    keep_sid=False 时先剔除 SID 再分词(SID 内部不泄漏为 a/b/数字)。
    词经 sys.intern 共享对象, 全量语料(数万篇)时显著省内存。"""
    tokens = []
    if keep_sid:
        pos = 0
        for m in SID_RE.finditer(text):
            tokens.extend(
                sys.intern(w.lower())
                for w in WORD_RE.findall(text[pos : m.start()])
            )
            tokens.append(m.group(0))
            pos = m.end()
        tokens.extend(
            sys.intern(w.lower()) for w in WORD_RE.findall(text[pos:])
        )
    else:
        text = SID_RE.sub(" ", text)
        tokens.extend(
            sys.intern(w.lower()) for w in WORD_RE.findall(text)
        )
    return tokens


def bigrams(tokens):
    return list(zip(tokens, tokens[1:]))


def norm_text(text):
    """规范化用于完全重复率判定: 去 SID、压缩空白、小写。"""
    return re.sub(r"\s+", " ", SID_RE.sub(" ", text)).strip().lower()


def first_sentence(text):
    s = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return s


def opener_template(text, generalize_sid=True):
    """首句模板: 小写、数字 -> <num>、去多余空白。
    generalize_sid=True 时 SID -> <sid>(默认, 与 teacher/模型语料
    同一套归并规则); False 时 SID 原样保留, 用于区分"连 SID 都一字
    不差"与"仅引用物品不同"两种首句重复。"""
    s = first_sentence(text).lower()
    if generalize_sid:
        s = SID_RE.sub("<sid>", s)
    s = re.sub(r"\d+(?:\.\d+)?", "<num>", s)
    s = re.sub(r"[^a-z0-9<> '.-]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .,-")


# --------------------------------------------------------------------------
# TF-IDF(无 sklearn 依赖; 全量语料也不建大矩阵)
# --------------------------------------------------------------------------
def idf_from_docs(doc_tokens):
    """返回 (idf 字典, 文档频率 df Counter)。
    idf = ln((1+N)/(1+df)) + 1(sklearn smooth_idf 配方)。"""
    df = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    n = len(doc_tokens)
    idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
    return idf, df


def dense_tfidf_rows(tok_lists, idf):
    """对给定(子)文档集构建 L2 行归一化的 TF-IDF 稠密矩阵。
    仅用于成对余弦, 规模受 pair_cap 限制。"""
    vocab = sorted({t for toks in tok_lists for t in toks})
    index = {t: i for i, t in enumerate(vocab)}
    mat = np.zeros((len(tok_lists), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(tok_lists):
        for t, c in Counter(toks).items():
            mat[i, index[t]] = (1.0 + math.log(c)) * idf.get(t, 1.0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def corpus_tfidf_summary(tok_lists, idf, top):
    """流式累计每个词在(行归一化后的)TF-IDF 权重之和与文档频率,
    返回 (Top 词表, df Counter)。不构建 N x V 大矩阵, 全量语料安全。"""
    weight_sum = {}
    df_counts = Counter()
    for toks in tok_lists:
        weights = {
            t: (1.0 + math.log(c)) * idf[t]
            for t, c in Counter(toks).items()
        }
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        for t, w in weights.items():
            weight_sum[t] = weight_sum.get(t, 0.0) + w / norm
            df_counts[t] += 1
    n = max(len(tok_lists), 1)
    nonstop = [t for t in weight_sum if t not in STOPWORDS]
    nonstop.sort(key=lambda t: -weight_sum[t])
    tfidf_top = [
        {"term": t, "mean_weight": round(weight_sum[t] / n, 4),
         "df": df_counts[t]}
        for t in nonstop[:top]
        if weight_sum[t] > 0
    ]
    return tfidf_top, df_counts


def shannon_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return h, (h / hmax if hmax > 0 else 0.0)


def percentile(values, q):
    return (
        float(np.percentile(np.asarray(values), q))
        if len(values)
        else 0.0
    )


# --------------------------------------------------------------------------
# 成对指标的 GPU 加速路径(torch 矩阵运算, 无 torch 或 OOM 时回退 CPU)
# --------------------------------------------------------------------------
def _torch_percentile(t, q):
    """GPU 上的分位数(nearest-rank): 先排序再取秩, 避开 torch.quantile
    的元素数限制(千万级成对值排序在 GPU 上是秒级)。"""
    import torch

    st = torch.sort(t).values
    k = min(
        max(int(round(q / 100.0 * (st.numel() - 1))), 0), st.numel() - 1
    )
    return st[k]


def _torch_pairwise_cosine(mat_np, device):
    """TF-IDF 矩阵的成对余弦。返回 (stats dict, iu, sims, )。
    mat_np: (n, V) L2 行归一化的 float32 矩阵。"""
    import torch

    m = torch.from_numpy(mat_np).to(device)
    sim = m @ m.T
    n = sim.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=device)
    sims = sim[iu[0], iu[1]]
    stats = {
        "n_pairs": int(sims.numel()),
        "mean": round(float(sims.mean()), 4),
        "median": round(float(_torch_percentile(sims, 50)), 4),
        "p95": round(float(_torch_percentile(sims, 95)), 4),
        "max": round(float(sims.max()), 4),
        "frac_gt_0.6": round(
            float((sims > 0.6).double().mean()), 4
        ),
        "frac_gt_0.8": round(
            float((sims > 0.8).double().mean()), 4
        ),
    }
    return stats, iu, sims


def _torch_pairwise_jaccard(shingle_sets, device):
    """词 3-gram 集合的成对 Jaccard: one-hot 矩阵相乘即交集大小,
    |并| = |A|+|B|-|交|。CPU 逐对集合运算的 O(n^2) 循环被两个
    矩阵乘法取代, 全量上万文档亦可秒级完成。"""
    import torch

    vocab = {}
    rows = []
    for s in shingle_sets:
        rows.append([vocab.setdefault(sh, len(vocab)) for sh in s])
    n, v = len(rows), len(vocab)
    onehot = torch.zeros(n, v, device=device)
    for i, ids in enumerate(rows):
        onehot[i, ids] = 1.0
    sizes = onehot.sum(dim=1)
    inter = onehot @ onehot.T
    iu = torch.triu_indices(n, n, offset=1, device=device)
    inter_v = inter[iu[0], iu[1]]
    union = (sizes[iu[0]] + sizes[iu[1]] - inter_v).clamp(min=1.0)
    jac = inter_v / union
    stats = {
        "n_pairs": int(jac.numel()),
        "mean": round(float(jac.mean()), 4),
        "median": round(float(_torch_percentile(jac, 50)), 4),
        "p95": round(float(_torch_percentile(jac, 95)), 4),
        "max": round(float(jac.max()), 4),
        "frac_gt_0.5": round(float((jac > 0.5).double().mean()), 4),
    }
    return stats


def resolve_device(spec):
    """auto: cuda > mps > cpu(torch 未安装则 cpu)。"""
    if spec != "auto":
        return spec
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------
# 单语料 × 单视角统计
# --------------------------------------------------------------------------
def lexical_block(tok_lists):
    """由 token 列表计算词汇类指标(词表/TTR/distinct/熵),
    返回 (指标 dict, unigram Counter, bigram Counter)。"""
    all_tokens = [t for toks in tok_lists for t in toks]
    all_bigrams = [b for toks in tok_lists for b in bigrams(toks)]
    uni_counts = Counter(all_tokens)
    bi_counts = Counter(all_bigrams)
    total_tokens = len(all_tokens)
    total_bigrams = len(all_bigrams)
    vocab_size = len(uni_counts)
    per_doc_d1 = [
        len(set(toks)) / len(toks) for toks in tok_lists if toks
    ]
    per_doc_d2 = [
        len(set(bigrams(toks))) / max(len(toks) - 1, 1)
        for toks in tok_lists
        if toks
    ]
    uni_h, uni_hn = shannon_entropy(uni_counts)
    bi_h, bi_hn = shannon_entropy(bi_counts)
    stats = {
        "vocab_size": vocab_size,
        "ttr": round(vocab_size / max(total_tokens, 1), 4),
        "distinct1_corpus": round(
            len(uni_counts) / max(total_tokens, 1), 4
        ),
        "distinct2_corpus": round(
            len(set(all_bigrams)) / max(total_bigrams, 1), 4
        ),
        "distinct1_doc_mean": round(
            float(np.mean(per_doc_d1)) if per_doc_d1 else 0.0, 4
        ),
        "distinct2_doc_mean": round(
            float(np.mean(per_doc_d2)) if per_doc_d2 else 0.0, 4
        ),
        "unigram_entropy_bits": round(uni_h, 3),
        "unigram_entropy_norm": round(uni_hn, 4),
        "bigram_entropy_bits": round(bi_h, 3),
        "bigram_entropy_norm": round(bi_hn, 4),
    }
    return stats, uni_counts, bi_counts


def jaccard_stats_from_tokens(sub_tok_lists, device):
    """对给定(子)文档 token 列表计算成对 3-gram Jaccard 统计。
    GPU 可用时走矩阵乘, 否则 CPU 双循环。"""
    shingles = []
    for t in sub_tok_lists:
        shingles.append(set(zip(t, t[1:], t[2:])) or {("<empty>",)})
    if device != "cpu":
        try:
            return _torch_pairwise_jaccard(shingles, device)
        except Exception as e:  # OOM / 算子缺失等
            print(f"[report] GPU Jaccard 失败({e}), 回退 CPU 循环")
    jacs = []
    for a in range(len(shingles)):
        sa = shingles[a]
        for b in range(a + 1, len(shingles)):
            sb = shingles[b]
            union = len(sa | sb)
            jacs.append(len(sa & sb) / union if union else 1.0)
    jacs = np.asarray(jacs)
    return {
        "n_pairs": int(len(jacs)),
        "mean": round(float(jacs.mean()), 4),
        "median": round(float(np.median(jacs)), 4),
        "p95": round(percentile(jacs, 95), 4),
        "max": round(float(jacs.max()), 4),
        "frac_gt_0.5": round(float((jacs > 0.5).mean()), 4),
    }


def analyze_variant(docs, keep_sid, top, pair_cap, rng, device="cpu"):
    """docs: 思维链文本列表。返回指标 dict。"""
    pairs = [
        (d, tokenize(d, keep_sid)) for d in docs
    ]
    pairs = [(d, t) for d, t in pairs if t]
    docs = [d for d, _ in pairs]
    tok_lists = [t for _, t in pairs]
    n_docs = len(tok_lists)

    # 长度
    lengths = [len(t) for t in tok_lists]
    char_lens = [len(d) for d in docs]

    # 词汇类指标: 全 token(含停用词, 与历史口径一致)
    lexical, uni_counts, bi_counts = lexical_block(tok_lists)
    # 词汇类指标: 去停用词(仅内容词)的对照版本
    content_tok_lists = [
        [t for t in toks if t not in STOPWORDS] for toks in tok_lists
    ]
    lexical_content, _, _ = lexical_block(content_tok_lists)

    # 词频(去停用词)与 TF-IDF Top 词
    content_counts = Counter(
        {t: c for t, c in uni_counts.items() if t not in STOPWORDS}
    )
    idf, _ = idf_from_docs(tok_lists)
    tfidf_top, df_counts = corpus_tfidf_summary(tok_lists, idf, top)
    freq_top = [
        {"term": t, "count": c, "df": df_counts.get(t, 0)}
        for t, c in content_counts.most_common(top)
    ]
    bigram_top = [
        {"bigram": f"{a} {b}", "count": c}
        for (a, b), c in bi_counts.most_common(top)
    ]

    # 完全重复率(规范化文本)
    norm_set = Counter(norm_text(d) for d in docs)
    n_unique = len(norm_set)
    dup_rate = 1.0 - n_unique / max(len(docs), 1)

    # 成对指标(采样至 pair_cap; TF-IDF 矩阵只在子集上构建)。
    # 余弦只用全 token 版(停用词承载句式模板信号); Jaccard 报
    # 全 token 与去停用词双口径
    idx = list(range(n_docs))
    if pair_cap and n_docs > pair_cap:
        idx = rng.sample(idx, pair_cap)
    cos_stats, jac_stats, jac_stats_content, top_pairs = {}, {}, {}, []
    if len(idx) >= 2:
        sub_toks = [tok_lists[i] for i in idx]
        sub = dense_tfidf_rows(sub_toks, idf)
        used_gpu = False
        if device != "cpu":
            try:
                import torch  # noqa: F401

                cos_stats, iu_t, sims_t = _torch_pairwise_cosine(
                    sub, device
                )
                vals, pos = torch.topk(sims_t, min(3, sims_t.numel()))
                for v, p in zip(vals.tolist(), pos.tolist()):
                    i, j = int(iu_t[0][p]), int(iu_t[1][p])
                    top_pairs.append(
                        {
                            "cosine": round(float(v), 4),
                            "a_user_doc": docs[idx[i]][:200],
                            "b_user_doc": docs[idx[j]][:200],
                        }
                    )
                used_gpu = True
            except Exception as e:  # OOM / 无 torch / 算子缺失等
                print(
                    f"[report] GPU 成对计算失败({e}), 回退 CPU 路径"
                )
        if not used_gpu:
            sim = sub @ sub.T
            iu = np.triu_indices(len(idx), k=1)
            sims = sim[iu].astype(np.float64)
            cos_stats = {
                "n_pairs": int(len(sims)),
                "mean": round(float(sims.mean()), 4),
                "median": round(float(np.median(sims)), 4),
                "p95": round(percentile(sims, 95), 4),
                "max": round(float(sims.max()), 4),
                "frac_gt_0.6": round(float((sims > 0.6).mean()), 4),
                "frac_gt_0.8": round(float((sims > 0.8).mean()), 4),
            }
            # 最相似样本对 Top-3(按余弦)
            pair_order = np.argsort(-sims)[:3]
            for p in pair_order:
                i, j = int(iu[0][p]), int(iu[1][p])
                top_pairs.append(
                    {
                        "cosine": round(float(sims[p]), 4),
                        "a_user_doc": docs[idx[i]][:200],
                        "b_user_doc": docs[idx[j]][:200],
                    }
                )
        jac_stats = jaccard_stats_from_tokens(sub_toks, device)
        jac_stats_content = jaccard_stats_from_tokens(
            [content_tok_lists[i] for i in idx], device
        )

    # 首句模板: 双口径 —— 泛化(SID-><sid>, 跨物品归并) 与
    # 原样(SID 原文保留, 仅统计连 SID 都一致的逐字重复)
    openers = [opener_template(d) for d in docs]
    opener_counts = Counter(openers)
    opener_top = [
        {"opener": o, "count": c, "frac": round(c / len(openers), 4)}
        for o, c in opener_counts.most_common(top)
    ]
    openers_strict = [opener_template(d, generalize_sid=False)
                      for d in docs]
    strict_counts = Counter(openers_strict)
    strict_top = [
        {"opener": o, "count": c, "frac": round(c / len(openers_strict), 4)}
        for o, c in strict_counts.most_common(top)
    ]

    # gzip 压缩率: 原文(全词)与去停用词拼接文本双口径
    blob = "\n".join(docs).encode("utf-8")
    gzip_ratio = (
        len(gzip.compress(blob, 9)) / len(blob) if blob else 0.0
    )
    content_blob = "\n".join(
        " ".join(toks) for toks in content_tok_lists
    ).encode("utf-8")
    gzip_ratio_content = (
        len(gzip.compress(content_blob, 9)) / len(content_blob)
        if content_blob
        else 0.0
    )

    return {
        "n_docs": n_docs,
        "n_docs_pairwise": len(idx),
        "length": {
            "words_mean": round(float(np.mean(lengths)), 1),
            "words_median": round(float(np.median(lengths)), 1),
            "words_p5": round(percentile(lengths, 5), 1),
            "words_p95": round(percentile(lengths, 95), 1),
            "chars_mean": round(float(np.mean(char_lens)), 1),
        },
        "lexical": lexical,
        "lexical_content": lexical_content,
        "tfidf_top_terms": tfidf_top,
        "freq_top_terms": freq_top,
        "top_bigrams": bigram_top,
        "duplication": {
            "exact_dup_rate": round(dup_rate, 4),
            "n_unique_docs": n_unique,
            "cosine": cos_stats,
            "jaccard3": jac_stats,
            "jaccard3_content": jac_stats_content,
        },
        "openers": {
            "unique_ratio": round(len(opener_counts) / len(openers), 4),
            "top1_frac": opener_top[0]["frac"] if opener_top else 0.0,
            "top5_frac": round(
                sum(o["frac"] for o in opener_top[:5]), 4
            ),
            "top": opener_top[:10],
        },
        "openers_strict": {
            "unique_ratio": round(
                len(strict_counts) / len(openers_strict), 4
            ),
            "top1_frac": strict_top[0]["frac"] if strict_top else 0.0,
            "top5_frac": round(
                sum(o["frac"] for o in strict_top[:5]), 4
            ),
            "top": strict_top[:10],
        },
        "gzip_ratio": round(gzip_ratio, 4),
        "gzip_ratio_content": round(gzip_ratio_content, 4),
        "most_similar_pairs": top_pairs,
    }


# --------------------------------------------------------------------------
# 报告渲染
# --------------------------------------------------------------------------
def _fmt_pct(x):
    return f"{100 * x:.1f}%"


def render_comparison(rows, variant, content=False):
    header = (
        "| 语料 | n | 词数均值 | TTR | dist-2(语料) | 完全重复率 |"
        " 余弦均值 | 余弦p95 | 余弦>0.8 | Jaccard均值 | 首句Top1 |"
        " 首句Top5 | gzip率 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    label = variant + (" · 去停用词" if content else "")
    lines = [f"\n### 汇总对比 · {label} 视角\n\n", header]
    for name, m in rows:
        lex = m["lexical_content"] if content else m["lexical"]
        jac_key = "jaccard3_content" if content else "jaccard3"
        jac = m["duplication"].get(jac_key, {})
        gzip_ratio = (
            m["gzip_ratio_content"] if content else m["gzip_ratio"]
        )
        cos = m["duplication"]["cosine"]
        lines.append(
            f"| {name} | {m['n_docs']} | "
            f"{m['length']['words_mean']} | {lex['ttr']:.4f} | "
            f"{lex['distinct2_corpus']:.4f} | "
            f"{_fmt_pct(m['duplication']['exact_dup_rate'])} | "
            f"{cos.get('mean', 0):.4f} | {cos.get('p95', 0):.4f} | "
            f"{_fmt_pct(cos.get('frac_gt_0.8', 0))} | "
            f"{jac.get('mean', 0):.4f} | "
            f"{_fmt_pct(m['openers']['top1_frac'])} | "
            f"{_fmt_pct(m['openers']['top5_frac'])} | "
            f"{gzip_ratio:.3f} |\n"
        )
    return "".join(lines)


def render_corpus_section(name, res, top_in_md):
    text = res["text"]
    raw = res["raw"]
    out = [f"\n## 语料: {name}\n"]
    meta_line = (
        f"- 来源: {res['meta'].get('kind', '未知')} | 采集 "
        f"{res['meta'].get('n_docs_collected', text['n_docs'])} 篇, "
        f"统计 {res['meta'].get('n_docs_analyzed', text['n_docs'])} 篇 | "
        f"思维链完整闭合率 "
        f"{_fmt_pct(res['meta'].get('think_closed_frac', 1.0))}\n"
    )
    out.append(meta_line)
    out.append(
        f"- 文档数 {text['n_docs']} | 词数均值 "
        f"{text['length']['words_mean']} (p5 {text['length']['words_p5']}, "
        f"p95 {text['length']['words_p95']}) | 词表 "
        f"{text['lexical']['vocab_size']} | TTR "
        f"{text['lexical']['ttr']:.4f} | gzip率 {text['gzip_ratio']:.3f}\n"
    )
    out.append(
        f"- 语料级 distinct-1/2 = "
        f"{text['lexical']['distinct1_corpus']:.4f} / "
        f"{text['lexical']['distinct2_corpus']:.4f}; 文档级均值 "
        f"{text['lexical']['distinct1_doc_mean']:.4f} / "
        f"{text['lexical']['distinct2_doc_mean']:.4f}; "
        f"unigram 熵 {text['lexical']['unigram_entropy_bits']:.1f} bits "
        f"(归一化 {text['lexical']['unigram_entropy_norm']:.3f}), "
        f"bigram 熵 {text['lexical']['bigram_entropy_bits']:.1f} bits "
        f"(归一化 {text['lexical']['bigram_entropy_norm']:.3f})\n"
    )
    dup = text["duplication"]
    out.append(
        f"- 完全重复率 {_fmt_pct(dup['exact_dup_rate'])}"
        f" ({dup['n_unique_docs']}/{text['n_docs']} 篇唯一);\n"
    )
    for label, v in (("text", text), ("raw", raw)):
        cos = v["duplication"]["cosine"]
        jac = v["duplication"]["jaccard3"]
        if cos:
            out.append(
                f"  - 成对余弦({label}): 均值 {cos['mean']:.4f}, 中位 "
                f"{cos['median']:.4f}, p95 {cos['p95']:.4f}, "
                f">0.6 {_fmt_pct(cos['frac_gt_0.6'])}, "
                f">0.8 {_fmt_pct(cos['frac_gt_0.8'])} | "
                f"3-gram Jaccard: 均值 {jac.get('mean', 0):.4f}, "
                f">0.5 {_fmt_pct(jac.get('frac_gt_0.5', 0))}\n"
            )
    # 去停用词对照(仅内容词; text 视角)
    lc = text["lexical_content"]
    jc = text["duplication"].get("jaccard3_content", {})
    out.append(
        f"- 去停用词对照(text, 仅内容词): 词表 {lc['vocab_size']} | "
        f"TTR {lc['ttr']:.4f} | distinct-1/2(语料) "
        f"{lc['distinct1_corpus']:.4f} / {lc['distinct2_corpus']:.4f} | "
        f"unigram 熵 {lc['unigram_entropy_bits']:.1f} bits "
        f"(归一化 {lc['unigram_entropy_norm']:.3f}), "
        f"bigram 熵 {lc['bigram_entropy_bits']:.1f} bits "
        f"(归一化 {lc['bigram_entropy_norm']:.3f}) | "
        f"3-gram Jaccard 均值 {jc.get('mean', 0):.4f} | "
        f"gzip率 {text.get('gzip_ratio_content', 0):.3f}\n"
    )
    op = text["openers"]
    out.append(
        f"- 首句模板: 唯一率 {_fmt_pct(op['unique_ratio'])}, "
        f"Top1 覆盖 {_fmt_pct(op['top1_frac'])}, "
        f"Top5 覆盖 {_fmt_pct(op['top5_frac'])}\n"
    )
    ops = text["openers_strict"]
    out.append(
        f"- 首句模板(原样, SID 不泛化, 即连 SID 都逐字一致的口径): "
        f"唯一率 {_fmt_pct(ops['unique_ratio'])}, "
        f"Top1 覆盖 {_fmt_pct(ops['top1_frac'])}, "
        f"Top5 覆盖 {_fmt_pct(ops['top5_frac'])}\n"
    )
    for o in op["top"][:5]:
        out.append(
            f"  - ({o['count']}x, {o['frac']:.1%}) {o['opener'][:120]}\n"
        )

    out.append(
        f"\n**TF-IDF 平均权重 Top{top_in_md} (text 视角)**\n\n"
        "| 词 | 平均TF-IDF权重 | 出现文档数 |\n|---|---|---|\n"
    )
    for t in text["tfidf_top_terms"][:top_in_md]:
        out.append(
            f"| {t['term']} | {t['mean_weight']:.4f} | {t['df']} |\n"
        )
    out.append(
        f"\n**原始词频 Top{top_in_md} (text 视角, 去停用词)**\n\n"
        "| 词 | 次数 | 文档数 |\n|---|---|---|\n"
    )
    for t in text["freq_top_terms"][:top_in_md]:
        out.append(f"| {t['term']} | {t['count']} | {t['df']} |\n")
    out.append(f"\n**Bigram 频率 Top{top_in_md} (text 视角)**\n\n")
    out.append(", ".join(
        f"{b['bigram']}({b['count']})"
        for b in text["top_bigrams"][:top_in_md]
    ) + "\n")
    out.append(f"\n**Bigram 频率 Top{top_in_md} (raw 视角, 含 SID)**\n\n")
    out.append(", ".join(
        f"`{b['bigram']}`({b['count']})"
        for b in raw["top_bigrams"][:top_in_md]
    ) + "\n")

    pairs = text["duplication"]["cosine"] and text["most_similar_pairs"]
    if pairs:
        out.append("\n**最相似样本对(余弦 Top, 截断 200 字符)**\n")
        for i, p in enumerate(pairs, 1):
            out.append(
                f"\n{i}. 余弦 {p['cosine']:.4f}\n"
                f"   - A: {p['a_user_doc']}\n"
                f"   - B: {p['b_user_doc']}\n"
            )
    return "".join(out)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="思维链多样性统计报告(详见模块 docstring)"
    )
    parser.add_argument(
        "--corpora", default=os.environ.get("CORPORA", ""),
        help="逗号分隔的语料名(默认 corpora/ 下全部)",
    )
    parser.add_argument("--top", type=int,
                        default=int(os.environ.get("TOP", 25)))
    parser.add_argument("--md-top", type=int,
                        default=int(os.environ.get("MD_TOP", 15)),
                        help="markdown 中展示的词条数")
    parser.add_argument("--pair-cap", type=int,
                        default=int(os.environ.get("PAIR_CAP", 600)),
                        help="成对余弦/Jaccard 的文档数上限(O(n^2) 计算), "
                             "0 = 不限制(全量 5 万篇时约 12 亿对, 慎用)")
    parser.add_argument("--max-docs", type=int,
                        default=int(os.environ.get("MAX_DOCS", 2000)),
                        help="每语料参与统计的文档数上限(超过按 seed 采样), "
                             "0 = 全量")
    parser.add_argument("--full", action="store_true",
                        default=bool(os.environ.get("FULL")),
                        help="全量模式: 等价于 --max-docs 0")
    parser.add_argument("--equalize", dest="equalize",
                        action="store_true", default=True,
                        help="(默认开)各语料采样到与最小语料相同的篇数, "
                             "保证 TTR/distinct/熵/gzip 等语料级指标可比")
    parser.add_argument("--no-equalize", dest="equalize", action="store_false",
                        help="关闭等量化, 按各自规模统计(语料级指标慎比)")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"),
                        help="成对指标计算设备: auto(默认, cuda>mps>cpu)"
                             "/cuda/mps/cpu。GPU 走 torch 矩阵乘, "
                             "PAIR_CAP=0 全量成对也秒级(A800 80GB 可容 "
                             "约 4 万篇; OOM 自动回退 CPU)")
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("SEED", 42)))
    parser.add_argument(
        "--out", default=os.environ.get("OUT", "analysis.json"),
        help="分析结果 JSON 文件名(如 analysis_temp.json, 与贪心版隔离);"
             " markdown 报告同名替换扩展名",
    )
    args = parser.parse_args()
    if args.full:
        args.max_docs = 0
    device = resolve_device(args.device)
    if device != "cpu":
        print(f"[report] 成对指标使用设备: {device}")

    from config import load_config
    from src.train.common import resolve_category

    cfg = load_config("sidreasoner")
    category = resolve_category(cfg["data"])
    root = os.path.join("results", "diversity", category)
    corpora_dir = os.path.join(root, "corpora")
    if not os.path.isdir(corpora_dir):
        raise SystemExit(
            f"未找到 {corpora_dir}, 先运行 "
            f"DATASET=<ds> .venv/bin/python -m src.analysis.collect_diversity_corpora"
        )

    names = (
        [n.strip() for n in args.corpora.split(",") if n.strip()]
        if args.corpora
        else sorted(
            f[:-6]
            for f in os.listdir(corpora_dir)
            if f.endswith(".jsonl")
        )
    )
    if not names:
        raise SystemExit("没有可用语料")

    manifest_path = os.path.join(root, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    results, errors = {}, []
    # 先全部载入, 便于跨语料等量化(见 --no-equalize 说明)
    loaded = {}
    for name in names:
        path = os.path.join(corpora_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            errors.append(f"语料不存在: {path}")
            continue
        docs, n_close = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                docs.append(rec["cot"])
                n_close += bool(rec.get("close_found", True))
        loaded[name] = (docs, n_close)

    # 语料级 TTR/distinct/熵/gzip 随语料大小单调变化, 规模不同的语料
    # 直接对比会失真; 默认把所有语料随机采样到与最小语料相同篇数
    sizes = {n: len(d) for n, (d, _) in loaded.items()}
    common_size = min(sizes.values()) if sizes else 0
    equalize = (
        args.equalize and len(set(sizes.values())) > 1
    )
    if equalize:
        print(
            f"[report] 等量化: 各语料采样到 {common_size} 篇"
            f"(最小语料规模; --no-equalize 可关闭)"
        )

    for name, (docs, n_close) in loaded.items():
        closed_frac = n_close / max(len(docs), 1)
        rng = random.Random(args.seed)
        if args.max_docs and len(docs) > args.max_docs:
            docs = rng.sample(docs, args.max_docs)
        if equalize and len(docs) > common_size:
            docs = rng.sample(docs, common_size)
        print(f"[report] {name}: 原始 {sizes[name]} 篇 -> 统计 {len(docs)} 篇")
        res = {
            "text": analyze_variant(docs, False, args.top,
                                    args.pair_cap, rng, device),
            "raw": analyze_variant(docs, True, args.top,
                                   args.pair_cap, rng, device),
        }
        res["meta"] = dict(manifest.get(name, {}))
        res["meta"]["n_docs_collected"] = sizes[name]
        res["meta"]["n_docs_analyzed"] = len(docs)
        res["meta"]["pairwise_device"] = device
        res["meta"]["think_closed_frac"] = round(closed_frac, 4)
        results[name] = res

    analysis_path = os.path.join(root, args.out)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md = [
        f"# SIDReasoner 思维链多样性报告\n",
        f"\n数据集: **{category}** | 生成时间: "
        f"{__import__('time').strftime('%Y-%m-%d %H:%M:%S')} | 统计参数: "
        f"top={args.top}, pair_cap={args.pair_cap}, "
        f"max_docs={args.max_docs}, seed={args.seed}, "
        f"equalize={'on' if args.equalize else 'off'}, "
        f"device={device}\n",
        f"\n- text 视角 = 剔除 SID token 后的自然语言; "
        f"raw 视角 = 保留 SID token。\n"
        f"- 关键读法: 成对余弦均值/完全重复率/首句 Top 覆盖越高、"
        f"distinct 与 gzip 率越低, 说明思维链越同质化。\n"
        f"- equalize=on 时各语料已采样到与最小语料相同的篇数"
        f"(语料级指标才可跨语料对比); 每个语料的采集总量见其小节。\n"
        f"- 词汇类指标/gzip/Jaccard 均有 含停用词(默认) 与 去停用词"
        f"(仅内容词, *_content / '去停用词' 对照) 两套口径。\n",
    ]
    md.append(render_comparison(
        [(n, results[n]["text"]) for n in results], "text"
    ))
    md.append(render_comparison(
        [(n, results[n]["raw"]) for n in results], "raw"
    ))
    md.append(render_comparison(
        [(n, results[n]["text"]) for n in results], "text", content=True
    ))
    for name in results:
        md.append(render_corpus_section(name, results[name], args.md_top))
    if errors:
        md.append("\n## 警告\n\n" + "\n".join(f"- {e}" for e in errors))

    report_path = os.path.join(root, os.path.splitext(args.out)[0] + ".md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"[report] 分析: {analysis_path}")
    print(f"[report] 报告: {report_path}")


if __name__ == "__main__":
    main()
