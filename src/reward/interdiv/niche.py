"""Niche(小众物品)多样性奖励 scorer。

设计(详见 docs/sidreasoner_inter_query_diversity.md 第 8 节): 对一条 rollout
CoT 计算 d = (c1 + c2 + c3) / 3, 三个分量:

- c1 信息提取比例: |niche(CoT SID) ∩ niche(history)| / |niche(history)|,
  集合语义(重复只计一次); 分子限定在用户历史内(堵死"编造冷门 SID"的
  reward hack); 历史无 niche 物品时 c1 = 0;
- c2 小众大众相似度: 提取 CoT 中包含 SID 的句子, 按"句内物品全 niche /
  全大众"分成两个去重集合(mixed 句从两侧剔除, 避免句子与自身算相似度),
  句子过 BGE 编码后取两集合全对余弦的均值 sim, c2 = 1 - sim;
  任一侧为空时 c2 = 0;
- c3 总结句相似度: 取 CoT 中最后一个 SID 所在句子之后的文本为总结段,
  与 8 个相似用户 Reference CoT 的总结段比相似度(度量可选
  sif=Arora et al. 2017 句向量 / tfidf=BM25 加权), c3 = 1 - mean cos;
  无有效 Reference 或总结段为空时 c3 = 0。

离线产物(make <dataset>-sidreasoner-niche-data,
src/data/sidreasoner_niche_preprocess.py, data/raw/Amazon/niche/<Category>/):
- popularity.json  物品流行度/niche 标记 + SID->item 映射(c1/c2 用)
- cot_lexstats.json CoT 语料词表 df/counts/域内停用词(c3 用)
- sif_words.npz    SIF 词向量/词频/公共成分(c3 method="sif" 用)
"""

import json
import math
import os
import re
from collections import Counter

import numpy as np

from src.analysis.diversity_analysis import STOPWORDS as BASE_STOPWORDS

#: SID token(任意层)与完整三元组(物品引用的最小单元)
SID_TOKEN_RE = re.compile(r"<[abc]_\d+>")
SID_TRIPLET_RE = re.compile(r"<a_\d+><b_\d+><c_\d+>")
#: 与 diversity_analysis 同口径的分词(小写、字母/数字、撇号缩写)
WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+")
#: 句子边界(句末标点或换行; 总结段 = 最后一个 SID 所在句子之后)
_SENT_BOUNDARY_RE = re.compile(r"[.!?\n]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize_content(text: str, stopwords=frozenset()) -> list:
    """去 SID -> WORD_RE 分词 -> 小写 -> 去停用词。"""
    text = SID_TOKEN_RE.sub(" ", text or "")
    return [w.lower() for w in WORD_RE.findall(text) if w.lower() not in stopwords]


def split_sentences(text: str) -> list:
    """按句末标点 + 换行切句(空句剔除)。"""
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


class BgeSentenceEncoder:
    """BGE 句子编码器(c2 的句子 embedding; CLS pooling + L2 归一化,
    与 src/data/rqvae_embed.py 同口径)。懒加载, 默认 CPU(reward 在
    verl driver 进程执行, 默认 cuda 会在 colocated GPU 上挤占显存)。"""

    def __init__(self, model: str = "BAAI/bge-base-en-v1.5",
                 device: str = "cpu", batch_size: int = 64):
        self.model_name = str(model)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self._model = None
        self._tokenizer = None

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        return self._model.config.hidden_size

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        )
        self._torch = torch

    def encode(self, texts: list) -> np.ndarray:
        """(N, dim) float32, 逐行 L2 归一化。"""
        self._ensure_loaded()
        torch = self._torch
        chunks = []
        for i in range(0, len(texts), self.batch_size):
            batch = self._tokenizer(
                texts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self._model(**batch).last_hidden_state[:, 0]
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            chunks.append(hidden.float().cpu())
        return torch.cat(chunks).numpy() if chunks else np.zeros((0, 1), dtype=np.float32)


def _cosine_dicts(a: dict, b: dict) -> float:
    common = a.keys() & b.keys()
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(a[k] * b[k] for k in common) / (na * nb)


def _cosine_vec(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class NicheDiversityScorer:
    """Niche 多样性 d(c1+c2+c3)/3(接口与 DiverBleu/NgramJaccard 的
    ``diversity`` 不同: 需要 CoT、Reference 列表与用户历史物品, 由
    InterQueryDiversityAddon 的 niche 分支批量调用)。

    Args:
        niche_dict_path: popularity.json(必须, make <ds>-sidreasoner-niche-data)。
        lexstats_path: cot_lexstats.json(必须)。
        summary_method: "sif"(默认, Arora et al. 2017 频率加权词向量平均
            + 减公共成分) | "tfidf"(BM25 idf + tf 饱和的稀疏向量余弦)。
        sif_path: sif_words.npz(method="sif" 时必须)。
        encoder: 句子编码器(c2); 缺省懒加载 BgeSentenceEncoder。
            测试/探针可注入 stub(只需提供 ``encode(texts)->ndarray``)。
        sif_a: SIF 权重常数 a(权重 = a/(a+p(w)))。
        bm25_k1 / bm25_b: tfidf 变体的 BM25 饱和参数(总结段长度相近,
            默认 b=0 不做长度归一)。
    """

    def __init__(
        self,
        niche_dict_path: str,
        lexstats_path: str,
        summary_method: str = "sif",
        sif_path: str = None,
        encoder=None,
        embed_model: str = "BAAI/bge-base-en-v1.5",
        embed_device: str = "cpu",
        embed_batch_size: int = 64,
        sif_a: float = 1e-3,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.0,
        sentence_cache_size: int = 100_000,
    ):
        self.summary_method = str(summary_method)
        if self.summary_method not in ("sif", "tfidf"):
            raise ValueError(
                f"未知 niche_summary_method: {summary_method!r}(sif | tfidf)"
            )
        self.sif_a = float(sif_a)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

        self._load_popularity(niche_dict_path)
        self._load_lexstats(lexstats_path)
        self._sif_emb = self._sif_p = self._sif_common = self._sif_index = None
        if self.summary_method == "sif":
            if not sif_path:
                raise ValueError('niche_summary_method="sif" 需要 sif_path。')
            self._load_sif(sif_path)

        self._encoder = encoder
        self._embed_cfg = {
            "model": str(embed_model),
            "device": str(embed_device),
            "batch_size": int(embed_batch_size),
        }
        self._sent_cache: dict = {}
        self._sent_cache_size = int(sentence_cache_size)
        self._summary_cache: dict = {}
        #: 最近一次 compute_many 的 batch 级统计(供 metrics 记录)
        self.last_stats: dict = {}

    # ------------------------------------------------------------------ #
    # 离线产物加载
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require(path: str, what: str):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"未找到 niche {what}: {path!r}。先运行 "
                "DATASET=<games|office|industrial> make <dataset>-sidreasoner-niche-data "
                "生成离线产物。"
            )

    def _load_popularity(self, path: str):
        self._require(path, "流行度字典(popularity.json)")
        with open(path, "r", encoding="utf-8") as f:
            pop = json.load(f)
        self.popularity_meta = pop.get("meta", {})
        self.sid_to_item = {k: int(v) for k, v in pop.get("sid_to_item", {}).items()}
        self.niche_items = {
            int(k) for k, v in pop.get("items", {}).items() if v.get("niche")
        }
        if not self.popularity_meta or not pop.get("items"):
            raise ValueError(f"流行度字典不完整(缺 meta/items): {path}")

    def _load_lexstats(self, path: str):
        self._require(path, "CoT 词表统计(cot_lexstats.json)")
        with open(path, "r", encoding="utf-8") as f:
            lex = json.load(f)
        self.lexstats_meta = lex.get("meta", {})
        self.n_docs = int(lex.get("n_docs", 0))
        self.avg_doc_len = float(lex.get("avg_doc_len", 0.0)) or None
        self.domain_stopwords = frozenset(lex.get("domain_stopwords", []))
        self.stopwords = BASE_STOPWORDS | self.domain_stopwords
        # BM25 概率化 idf; 未登录词取 df=0 的最大 idf(当最稀有词,
        # 与 diversity_analysis 的 idf.get(t, 1.0) 方向相反)
        n = max(self.n_docs, 1)
        self._idf = {
            w: math.log(1.0 + (n - d + 0.5) / (d + 0.5))
            for w, d in lex.get("df", {}).items()
        }
        self.max_idf = math.log(1.0 + (n + 0.5) / 0.5)

    def _load_sif(self, path: str):
        self._require(path, "SIF 词向量(sif_words.npz)")
        data = np.load(path, allow_pickle=False)
        vocab = [str(w) for w in data["vocab"].tolist()]
        self._sif_emb = data["embeddings"].astype(np.float32)
        self._sif_p = data["p"].astype(np.float64)
        self._sif_common = data["common"].astype(np.float64)
        self._sif_index = {w: i for i, w in enumerate(vocab)}
        if not (len(vocab) == len(self._sif_p) == self._sif_emb.shape[0]):
            raise ValueError(f"sif_words.npz 内部长度不一致: {path}")

    # ------------------------------------------------------------------ #
    # CoT 解析
    # ------------------------------------------------------------------ #

    def resolve_items(self, cot: str) -> set:
        """CoT 中完整 SID 三元组 -> 物品 id 集合(幻觉 SID 丢弃)。"""
        items = set()
        for sid in SID_TRIPLET_RE.findall(cot or ""):
            item = self.sid_to_item.get(sid)
            if item is not None:
                items.add(item)
        return items

    def classify_sentences(self, cot: str):
        """返回 (niche 句集合, popular 句集合)。

        - 句子 = 按 [.!?]/换行切分后的最小单位, 需包含 >=1 个完整 SID 三元组;
        - 句内物品全部 niche -> niche 侧; 全部大众 -> popular 侧;
          mixed(同句混含) -> 两侧都剔除(避免与自身算 sim=1 拉低 reward);
        - 集合语义: 同一句子多次出现只计一次(按压缩空白后的文本去重)。
        """
        niche_sents, pop_sents = set(), set()
        for sent in split_sentences(cot or ""):
            if not SID_TRIPLET_RE.search(sent):
                continue
            items = {
                self.sid_to_item.get(s)
                for s in SID_TRIPLET_RE.findall(sent)
            }
            items.discard(None)
            if not items:
                continue
            key = " ".join(sent.split())
            n_niche = sum(1 for i in items if i in self.niche_items)
            if n_niche == len(items):
                niche_sents.add(key)
            elif n_niche == 0:
                pop_sents.add(key)
        return niche_sents, pop_sents

    @staticmethod
    def extract_summary(cot: str) -> str:
        """最后一个 SID 所在句子之后的全部文本(不含该句)。

        - 无 SID 或其后没有句界 -> ""(c3 取 0);
        - "最后一个 SID" 按任意层 token 计(截断的半截 SID 也算边界)。
        """
        last = None
        for m in SID_TOKEN_RE.finditer(cot or ""):
            last = m
        if last is None:
            return ""
        boundary = _SENT_BOUNDARY_RE.search(cot, last.end())
        if boundary is None:
            return ""
        return cot[boundary.end() :].strip()

    # ------------------------------------------------------------------ #
    # 向量化
    # ------------------------------------------------------------------ #

    def _tfidf_vector(self, tokens: list):
        """BM25 加权稀疏向量(idf 概率化, tf 饱和); 空 -> None。"""
        if not tokens:
            return None
        counts = Counter(tokens)
        length = len(tokens)
        avg = self.avg_doc_len or float(length)
        vec = {}
        for w, tf in counts.items():
            idf = self._idf.get(w, self.max_idf)
            denom = tf + self.bm25_k1 * (
                1.0 - self.bm25_b + self.bm25_b * length / avg
            )
            vec[w] = idf * tf * (self.bm25_k1 + 1.0) / denom
        return vec or None

    def _sif_vector(self, tokens: list):
        """SIF 句向量: a/(a+p(w)) 加权词向量平均 - 公共成分投影; 空 -> None。"""
        idxs = [self._sif_index[w] for w in tokens if w in self._sif_index]
        if not idxs:
            return None
        weights = self.sif_a / (self.sif_a + self._sif_p[idxs])
        vec = (self._sif_emb[idxs].astype(np.float64) * weights[:, None]).sum(
            axis=0
        ) / len(idxs)
        vec = vec - float(vec @ self._sif_common) * self._sif_common
        norm = float(np.linalg.norm(vec))
        return None if norm == 0.0 else vec / norm

    def _summary_vector(self, text: str):
        """总结段 -> 向量(tfidf: dict; sif: ndarray), 带缓存。"""
        if text in self._summary_cache:
            return self._summary_cache[text]
        tokens = tokenize_content(text, self.stopwords)
        if self.summary_method == "tfidf":
            vec = self._tfidf_vector(tokens)
        else:
            vec = self._sif_vector(tokens)
        if len(self._summary_cache) > self._sent_cache_size:
            self._summary_cache.clear()
        self._summary_cache[text] = vec
        return vec

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = BgeSentenceEncoder(**self._embed_cfg)
        return self._encoder

    def encode_sentences(self, texts) -> dict:
        """批量句子 embedding(去重 + 缓存), 返回 {text: ndarray}。"""
        out = {}
        missing = []
        for t in texts:
            if t in self._sent_cache:
                out[t] = self._sent_cache[t]
            else:
                missing.append(t)
        if missing:
            vecs = self._get_encoder().encode(missing)
            for t, v in zip(missing, vecs):
                v = np.asarray(v, dtype=np.float64)
                norm = float(np.linalg.norm(v))
                v = v / norm if norm > 0 else v
                if len(self._sent_cache) > self._sent_cache_size:
                    self._sent_cache.clear()
                self._sent_cache[t] = v
                out[t] = v
        return out

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #

    def compute_one(self, cot: str, refs: list, history_items) -> tuple:
        return self.compute_many(
            [{"cot": cot, "refs": refs, "history_items": history_items}]
        )[0]

    def compute_many(self, items: list) -> list:
        """批量计算 niche 多样性。

        Args:
            items: [{"cot": str, "refs": [str, ...](可为空, c3 取 0),
                     "history_items": array-like[int]}]

        Returns:
            [(d: float, diag: dict)] 与输入对齐; diag 含 niche_c1/c2/c3、
            各分量 defined 标志与计数。同时更新 self.last_stats。
        """
        parsed = []
        all_sentences: set = set()
        all_summaries: set = set()
        for it in items:
            cot = it["cot"]
            # 注意: history_items 可能是 np.ndarray, 不能写 `or []`(多元素
            # 数组的真值判定会抛 ValueError)
            hist = it.get("history_items")
            hist = list(hist) if hist is not None else []
            niche_hist = {int(x) for x in hist}
            niche_hist &= self.niche_items
            mentioned = self.resolve_items(cot) & niche_hist
            c1 = len(mentioned) / len(niche_hist) if niche_hist else 0.0
            niche_sents, pop_sents = self.classify_sentences(cot)
            summary = self.extract_summary(cot)
            ref_summaries = [self.extract_summary(r) for r in it.get("refs") or []]
            parsed.append(
                {
                    "c1": c1,
                    "defined1": bool(niche_hist),
                    "hist_n": len(niche_hist),
                    "mentioned_n": len(mentioned),
                    "niche_sents": niche_sents,
                    "pop_sents": pop_sents,
                    "summary": summary,
                    "ref_summaries": ref_summaries,
                }
            )
            all_sentences |= niche_sents | pop_sents
            all_summaries.add(summary)
            all_summaries.update(s for s in ref_summaries if s)

        sent_vecs = self.encode_sentences(sorted(all_sentences)) if all_sentences else {}
        summ_vecs = {
            s: (self._summary_vector(s) if s else None) for s in all_summaries
        }

        results = []
        c1s, c2s, c3s = [], [], []
        d1s, d2s, d3s = [], [], []
        for p in parsed:
            # ---- c2: 1 - mean cosine(niche 句 x popular 句) ----
            if p["niche_sents"] and p["pop_sents"]:
                mat_n = np.stack([sent_vecs[s] for s in sorted(p["niche_sents"])])
                mat_p = np.stack([sent_vecs[s] for s in sorted(p["pop_sents"])])
                sims = mat_n @ mat_p.T
                c2 = float(np.clip(1.0 - float(sims.mean()), 0.0, 1.0))
                defined2 = True
            else:
                c2, defined2 = 0.0, False

            # ---- c3: 1 - mean cosine(自己总结段, 各邻居总结段) ----
            own = summ_vecs.get(p["summary"]) if p["summary"] else None
            ref_vecs = [summ_vecs[s] for s in p["ref_summaries"]]
            ref_vecs = [v for v in ref_vecs if v is not None]
            if own is not None and ref_vecs:
                if isinstance(own, dict):
                    sims = [_cosine_dicts(own, v) for v in ref_vecs]
                else:
                    sims = [_cosine_vec(own, v) for v in ref_vecs]
                c3 = float(np.clip(1.0 - float(np.mean(sims)), 0.0, 1.0))
                defined3 = True
            else:
                c3, defined3 = 0.0, False

            d = (p["c1"] + c2 + c3) / 3.0
            results.append(
                (
                    float(d),
                    {
                        "niche_c1": float(p["c1"]),
                        "niche_c2": float(c2),
                        "niche_c3": float(c3),
                        "niche_c1_defined": int(p["defined1"]),
                        "niche_c2_defined": int(defined2),
                        "niche_c3_defined": int(defined3),
                        "niche_hist_count": int(p["hist_n"]),
                        "niche_mentioned_count": int(p["mentioned_n"]),
                    },
                )
            )
            c1s.append(p["c1"]); c2s.append(c2); c3s.append(c3)
            d1s.append(p["defined1"]); d2s.append(defined2); d3s.append(defined3)

        n = max(len(parsed), 1)
        self.last_stats = {
            "c1_mean": float(np.mean(c1s)) if parsed else 0.0,
            "c2_mean": float(np.mean(c2s)) if parsed else 0.0,
            "c3_mean": float(np.mean(c3s)) if parsed else 0.0,
            "c1_defined_frac": float(np.mean(d1s)) if parsed else 0.0,
            "c2_defined_frac": float(np.mean(d2s)) if parsed else 0.0,
            "c3_defined_frac": float(np.mean(d3s)) if parsed else 0.0,
            "hist_niche_mean": float(np.mean([p["hist_n"] for p in parsed])) if parsed else 0.0,
            "n_scored": len(parsed),
        }
        return results
