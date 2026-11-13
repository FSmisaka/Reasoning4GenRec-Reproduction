"""
思维链多样性分析(单文件输入)

对**一份**思考模式评测结果 JSON(逐条记录、含 cot 字段, 由
src/eval/sidreasoner_eval_think.py 产出并按 results/sidreasoner/
<Dataset>/<variant>/<step>.json 两级路由)做多样性统计, 结果写入
镜像路径::

    输入  results/sidreasoner/Games/think_steprule/step_1900.json
    输出  results/diversity/Games/think_steprule/step_1900.json

输出即一个 JSON 文件, 结构 {"text": {...}, "raw": {...}, "meta": {...}}:
  - text 视角: 剔除 SID token 后的自然语言(检验"推理话术是否套路化");
  - raw  视角: SID token 作为一个词保留(含物品引用的整体重复度);
  - meta: 输入路径 / 文档数 / 闭合率 / 统计参数。

统计项(每视角):
  1. TF-IDF 词频(sublinear tf + smooth idf + L2; 平均权重 Top 词,
     另给去停用词后的原始词频 Top)与 bigram 频率 Top;
  2. 词汇多样性: 词表 / TTR / 语料级与文档级 distinct-1,2 /
     unigram,bigram 熵(均有含停用词 lexical 与去停用词 lexical_content
     两套口径);
  3. 重复度: 完全重复率; 成对 TF-IDF 余弦与成对 3-gram Jaccard
     (后者亦有去停用词版 jaccard3_content);
  4. 首句模板集中度(SID 泛化 openers 与原样 openers_strict 双口径);
  5. gzip 压缩率(原文与去停用词拼接双口径);
  6. 最相似样本对示例(余弦 Top-3)。

用法:
  .venv/bin/python -m src.analysis.diversity_analysis \
      results/sidreasoner/Games/think_steprule/step_1900.json
  可选: --pair-cap 600 --device auto --top 25 --seed 42 --out <显式输出路径>
  (输入必须在 results/sidreasoner/ 下; 否则需显式 --out。
   注意 _summary.json 是跨 seed 聚合、无逐条 cot, 不能作为输入——
   需要单 seed 明细时置 config evaluation.save_per_seed_results: true)

成对指标在 GPU 可用时走 torch 矩阵乘(cuda>mps>cpu 自动检测,
DEVICE=cuda 强制), PAIR_CAP=0 可取消上限(A800 80GB 可容约 4 万篇)。
"""

import argparse
import gzip
import json
import math
import os
import random
import re
import sys
import time
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

THINK_ROOT = os.path.join("results", "sidreasoner")
DIVERSITY_ROOT = os.path.join("results", "diversity")


# --------------------------------------------------------------------------
# 清洗与分词
# --------------------------------------------------------------------------
def strip_think_tags(text):
    """剥掉模型生成思维链的 <think>/</think> 标签; 空内容返回 None。"""
    if text is None or (isinstance(text, float) and pd_isna(text)):
        return None
    text = str(text).strip()
    if not text:
        return None
    if text.startswith("<think>"):
        text = text[len("<think>") :]
    text = text.replace("<think>", "").replace("</think>", "")
    text = text.strip()
    return text or None


def pd_isna(x):
    try:
        import pandas as pd

        return bool(pd.isna(x))
    except Exception:
        return x != x


def tokenize(text, keep_sid):
    """keep_sid=True 时 SID 作为一个词保留; False 时先剔除 SID 再分词。
    词经 sys.intern 共享对象, 大语料省内存。"""
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
    return re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]


def opener_template(text, generalize_sid=True):
    """首句模板: 小写、数字-><num>、去多余空白; generalize_sid=True 时
    SID-><sid>(跨物品归并), False 时 SID 原样(逐字一致口径)。"""
    s = first_sentence(text).lower()
    if generalize_sid:
        s = SID_RE.sub("<sid>", s)
    s = re.sub(r"\d+(?:\.\d+)?", "<num>", s)
    s = re.sub(r"[^a-z0-9<> '.-]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .,-")


# --------------------------------------------------------------------------
# TF-IDF 与熵
# --------------------------------------------------------------------------
def shannon_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    h = -sum((c / total) * math.log2(c / total) for c in counts.values())
    hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return h, (h / hmax if hmax > 0 else 0.0)


def idf_from_docs(doc_tokens):
    """(idf 字典, df Counter); idf = ln((1+N)/(1+df)) + 1。"""
    df = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    n = len(doc_tokens)
    return {
        t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()
    }, df


def dense_tfidf_rows(tok_lists, idf):
    """(子)文档集的 L2 行归一化 TF-IDF 稠密矩阵, 仅用于成对余弦。"""
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
    """流式累计各词(行归一化后)TF-IDF 权重和与文档频率, 返回 Top 词表。"""
    weight_sum, df_counts = {}, Counter()
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
    return (
        [
            {
                "term": t,
                "mean_weight": round(weight_sum[t] / n, 4),
                "df": df_counts[t],
            }
            for t in nonstop[:top]
            if weight_sum[t] > 0
        ],
        df_counts,
    )


def percentile(values, q):
    return (
        float(np.percentile(np.asarray(values), q))
        if len(values)
        else 0.0
    )


# --------------------------------------------------------------------------
# 成对指标的 GPU 加速路径
# --------------------------------------------------------------------------
def _frac_gt(t, thr):
    """(t > thr) 的占比; int64 计数再除, 兼容 MPS(无 float64)且大语料精确。"""
    n = t.numel()
    return round(float((t > thr).sum().item()) / n, 4) if n else 0.0


def _torch_percentile(t, q):
    """与 np.percentile(linear)一致的插值分位数(GPU 上排序实现,
    避开 torch.quantile 的元素数限制)。"""
    import torch

    st = torch.sort(t).values
    n = st.numel()
    if n == 1:
        return st[0]
    pos = q / 100.0 * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return st[lo] + (st[hi] - st[lo]) * frac


def _torch_pairwise_cosine(mat_np, device):
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
        "frac_gt_0.6": _frac_gt(sims, 0.6),
        "frac_gt_0.8": _frac_gt(sims, 0.8),
    }
    return stats, iu, sims


def _torch_pairwise_jaccard(shingle_sets, device):
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
        "frac_gt_0.5": _frac_gt(jac, 0.5),
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
# 单视角统计
# --------------------------------------------------------------------------
def lexical_block(tok_lists):
    """词汇类指标(词表/TTR/distinct/熵), 返回 (指标 dict, 两个 Counter)。"""
    all_tokens = [t for toks in tok_lists for t in toks]
    all_bigrams = [b for toks in tok_lists for b in bigrams(toks)]
    uni_counts, bi_counts = Counter(all_tokens), Counter(all_bigrams)
    total_tokens, total_bigrams = len(all_tokens), len(all_bigrams)
    vocab_size = len(uni_counts)
    per_d1 = [len(set(t)) / len(t) for t in tok_lists if t]
    per_d2 = [
        len(set(bigrams(t))) / max(len(t) - 1, 1)
        for t in tok_lists
        if t
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
            float(np.mean(per_d1)) if per_d1 else 0.0, 4
        ),
        "distinct2_doc_mean": round(
            float(np.mean(per_d2)) if per_d2 else 0.0, 4
        ),
        "unigram_entropy_bits": round(uni_h, 3),
        "unigram_entropy_norm": round(uni_hn, 4),
        "bigram_entropy_bits": round(bi_h, 3),
        "bigram_entropy_norm": round(bi_hn, 4),
    }
    return stats, uni_counts, bi_counts


def jaccard_stats_from_tokens(sub_tok_lists, device):
    """成对 3-gram Jaccard; GPU 可用时走矩阵乘, 否则 CPU 双循环。"""
    shingles = []
    for t in sub_tok_lists:
        shingles.append(set(zip(t, t[1:], t[2:])) or {("<empty>",)})
    if device != "cpu":
        try:
            return _torch_pairwise_jaccard(shingles, device)
        except Exception as e:
            print(f"[diversity] GPU Jaccard 失败({e}), 回退 CPU 循环")
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
    """docs: 思维链文本列表。返回该视角的全部指标 dict。"""
    pairs = [(d, tokenize(d, keep_sid)) for d in docs]
    pairs = [(d, t) for d, t in pairs if t]
    docs = [d for d, _ in pairs]
    tok_lists = [t for _, t in pairs]
    n_docs = len(tok_lists)

    lengths = [len(t) for t in tok_lists]
    char_lens = [len(d) for d in docs]

    lexical, uni_counts, bi_counts = lexical_block(tok_lists)
    content_tok_lists = [
        [t for t in toks if t not in STOPWORDS] for toks in tok_lists
    ]
    lexical_content, _, _ = lexical_block(content_tok_lists)

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

    # 完全重复率
    norm_set = Counter(norm_text(d) for d in docs)
    n_unique = len(norm_set)
    dup_rate = 1.0 - n_unique / max(len(docs), 1)

    # 成对指标(采样至 pair_cap; 余弦只用全 token——停用词承载句式信号;
    # Jaccard 双口径)
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
                import torch

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
            except Exception as e:
                print(f"[diversity] GPU 成对计算失败({e}), 回退 CPU")
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

    # 首句模板(泛化 + 原样双口径)
    openers = [opener_template(d) for d in docs]
    opener_counts = Counter(openers)
    opener_top = [
        {"opener": o, "count": c, "frac": round(c / len(openers), 4)}
        for o, c in opener_counts.most_common(top)
    ]
    openers_strict = [opener_template(d, False) for d in docs]
    strict_counts = Counter(openers_strict)
    strict_top = [
        {"opener": o, "count": c, "frac": round(c / len(openers_strict), 4)}
        for o, c in strict_counts.most_common(top)
    ]

    # gzip 压缩率(原文 + 去停用词拼接)
    blob = "\n".join(docs).encode("utf-8")
    gzip_ratio = len(gzip.compress(blob, 9)) / len(blob) if blob else 0.0
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
# 输入解析与输出路由
# --------------------------------------------------------------------------
def load_think_records(path):
    """读取一份思考模式评测结果 JSON, 返回 (有效 cot 列表, 闭合率)。
    兼容逐条记录的列表(标准)与包一层的 {"records": [...]}。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            data = data["records"]
        else:
            raise SystemExit(
                f"[diversity] 输入是 dict 而非逐条记录的列表"
                f"(疑似 _summary 聚合文件, 不含逐条 cot):\n  {path}\n"
                f"  请改用单 seed 明细或贪心结果文件; 多 seed 评测需要"
                f"明细时置 config evaluation.save_per_seed_results: true"
            )
    docs, n_close = [], 0
    for rec in data:
        cot = strip_think_tags(rec.get("cot", ""))
        if cot is None:
            continue
        docs.append(cot)
        n_close += bool(rec.get("close_found", True))
    if not docs:
        raise SystemExit(f"[diversity] 输入中没有有效 cot 字段: {path}")
    return docs, n_close / len(docs)


def mirror_output_path(input_path):
    """results/sidreasoner/<D>/<variant>/<f>.json ->
    results/diversity/<D>/<variant>/<f>.json"""
    rel = os.path.relpath(os.path.abspath(input_path),
                          os.path.abspath(THINK_ROOT))
    if rel.startswith(".."):
        raise SystemExit(
            f"[diversity] 输入不在 {THINK_ROOT}/ 下: {input_path}\n"
            f"  请用 --out 显式指定输出路径"
        )
    return os.path.join(DIVERSITY_ROOT, rel)


def main():
    parser = argparse.ArgumentParser(
        description="单文件思维链多样性分析(详见模块 docstring)"
    )
    parser.add_argument(
        "input", nargs="?", default=os.environ.get("INPUT", ""),
        help="思考模式评测结果 JSON 路径(或用环境变量 INPUT)",
    )
    parser.add_argument(
        "--out", default=os.environ.get("OUT", ""),
        help="显式输出路径(默认镜像输入路径到 results/diversity/ 下)",
    )
    parser.add_argument("--top", type=int,
                        default=int(os.environ.get("TOP", 25)))
    parser.add_argument("--pair-cap", type=int,
                        default=int(os.environ.get("PAIR_CAP", 600)),
                        help="成对余弦/Jaccard 文档数上限, 0=不限制")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"),
                        help="成对指标计算设备: auto/cuda/mps/cpu")
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("SEED", 42)))
    args = parser.parse_args()
    if not args.input:
        raise SystemExit("用法: python -m src.analysis.diversity_analysis "
                         "<results/sidreasoner/.../xxx.json>")

    if not os.path.exists(args.input):
        raise SystemExit(f"[diversity] 输入不存在: {args.input}")
    device = resolve_device(args.device)
    docs, closed_frac = load_think_records(args.input)
    print(f"[diversity] {args.input}: {len(docs)} 条有效思维链"
          f"(闭合率 {closed_frac:.1%}), device={device}")

    rng = random.Random(args.seed)
    result = {
        "text": analyze_variant(docs, False, args.top,
                                args.pair_cap, rng, device),
        "raw": analyze_variant(docs, True, args.top,
                               args.pair_cap, rng, device),
        "meta": {
            "input": args.input,
            "n_docs": len(docs),
            "think_closed_frac": round(closed_frac, 4),
            "top": args.top,
            "pair_cap": args.pair_cap,
            "device": device,
            "seed": args.seed,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    out = args.out or mirror_output_path(args.input)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[diversity] 分析已写入 {out}")


if __name__ == "__main__":
    main()
