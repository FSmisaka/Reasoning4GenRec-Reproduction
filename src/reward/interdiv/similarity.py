"""n-gram 集合 Jaccard 相似度(跨 Query 多样性度量的 BLEU 替代品)。

    sim(a, b) = |G_n(a) ∩ G_n(b)| / |G_n(a) ∪ G_n(b)|
    diversity = 1 - sim

- G_n(text): text 前 div_len 字符 whitespace 分词后的 n-gram **集合**;
- 与 set 语义一致: 重复 n-gram 只计一次(不看频次);
- 接口与 :class:`src.reward.interdiv.bleu.DiverBleu` 一致
  (``similarity`` / ``diversity``), 可在编排器中直接替换;
- 缓存有界(lru 4096 条文本), 长训练不累积内存。
"""

from functools import lru_cache

NGRAM_DIV_LEN = 1000


class NgramJaccard:
    """n-gram 集合 Jaccard 相似度。

    Args:
        n: n-gram 阶数(>=1; 默认 1 = unigram 集合)。
        div_len: 参与比较的文本前缀字符数(与 BLEU 口径一致, 默认 1000)。
    """

    def __init__(self, n: int = 1, div_len: int = NGRAM_DIV_LEN):
        self.n = int(n)
        if self.n < 1:
            raise ValueError(f"n 必须 >= 1, 当前 {n}")
        self.div_len = int(div_len)

    @lru_cache(maxsize=4096)
    def _grams(self, text: str) -> frozenset:
        tokens = tuple(text[: self.div_len].split())
        return frozenset(
            tuple(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)
        )

    def similarity(self, a: str, b: str) -> float:
        ga, gb = self._grams(a), self._grams(b)
        union = len(ga | gb)
        if union == 0:
            return 0.0
        return len(ga & gb) / union

    def diversity(self, a: str, b: str) -> float:
        """1 - Jaccard(两文本差异度, 与 BLEU 版的 diversity 同义)。"""
        return 1.0 - self.similarity(a, b)
