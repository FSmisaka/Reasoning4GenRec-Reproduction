"""DIVER 风格的 Textual Diversity BLEU。

逐点复刻 DIVER 官方实现(NJU-RL/DIVER, verl/div_src/diversity_metric.py
``calculate_belu_matrix``)的度量口径:

- tokenization: 纯 whitespace 切分(``text.split()``);
- 比较前把文本截断到前 ``div_len`` 个字符(默认 1000);
- ``nltk.translate.bleu_score.sentence_bleu``, weights=(0.02, 0.1, 0.15,
  0.25, 0.38)(5-gram 加权几何平均, 权重和 0.9 —— 与官方代码一致, 不
  重新归一);
- 无平滑(method0): 任一阶 clipped precision 为 0 -> BLEU = 0;
- 相似度取双向对称平均: sim(a,b) = (BLEU(ref=a,cand=b) + BLEU(ref=b,cand=a)) / 2。

实现为纯 Python + n-gram Counter 缓存, 不依赖 nltk(服务器 RL 环境无需
安装); 数学语义与 nltk sentence_bleu(默认无平滑)一致:

    p_n  = clipped_ngram_matches / total_candidate_ngrams   (n = 1..5)
    BP   = 1                                   if len(cand) > len(ref)
           exp(1 - len(ref)/len(cand))         if len(cand) > 0
           0                                   if len(cand) == 0
    BLEU = BP * exp( sum_n w_n * ln p_n )      (若任一 p_n == 0 或分母
                                                为 0, 直接取 0)

nltk 在候选过短(不足 5-gram)时同样会因零精度退化, DIVER 的数学推理
域数据未触达该分支; 此处显式定义为 0, 行为一致且不抛异常。
"""

from collections import Counter
from functools import lru_cache

#: DIVER 官方权重(5-gram, 和为 0.9, 保持原样)
DIVER_BLEU_WEIGHTS = (0.02, 0.1, 0.15, 0.25, 0.38)
DIVER_DIV_LEN = 1000


def _ngram_counts(tokens: tuple, n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


class _TextStats:
    """单个文本(截断+分词后)的 n-gram 统计缓存。"""

    __slots__ = ("tokens", "ngrams", "length")

    def __init__(self, tokens: tuple, max_order: int):
        self.tokens = tokens
        self.length = len(tokens)
        self.ngrams = [_ngram_counts(tokens, n) for n in range(1, max_order + 1)]


class DiverBleu:
    """带缓存的 DIVER BLEU 打分器。

    Args:
        weights: n-gram 几何平均权重(默认 DIVER 官方 5-gram 权重)。
        div_len: 参与比较的文本前缀字符数(默认 1000, 官方默认)。
    """

    def __init__(self, weights=DIVER_BLEU_WEIGHTS, div_len: int = DIVER_DIV_LEN):
        w = tuple(float(x) for x in weights)
        if len(w) < 1:
            raise ValueError("weights 不能为空。")
        self.weights = w
        self.max_order = len(w)
        self.div_len = int(div_len)

    # ------------------------------------------------------------------ #
    # 缓存(有界 lru: 每个 step 的 rollout 文本数 ~B*G, 4096 上限足够
    # 覆盖单 step 全量并跨 step 复用; 避免无界 dict 在长训练中累积内存)
    # ------------------------------------------------------------------ #

    def _stats(self, text: str) -> "_TextStats":
        return self._stats_cached(text)

    @lru_cache(maxsize=4096)
    def _stats_cached(self, text: str) -> "_TextStats":
        truncated = text[: self.div_len]
        tokens = tuple(truncated.split())
        return _TextStats(tokens, self.max_order)

    # ------------------------------------------------------------------ #
    # BLEU
    # ------------------------------------------------------------------ #

    def bleu(self, reference: str, candidate: str) -> float:
        """BLEU(candidate | reference), nltk sentence_bleu 无平滑语义。"""
        ref = self._stats(reference)
        cand = self._stats(candidate)

        if cand.length == 0 or ref.length == 0:
            return 0.0

        import math

        log_sum = 0.0
        for n, w in enumerate(self.weights, start=1):
            cand_ngrams = cand.ngrams[n - 1]
            ref_ngrams = ref.ngrams[n - 1]
            total = cand.length - n + 1
            if total <= 0:
                return 0.0
            matches = sum(
                (cand_ngrams[g] if cand_ngrams[g] < ref_ngrams[g] else ref_ngrams[g])
                for g in cand_ngrams
                if g in ref_ngrams
            )
            if matches <= 0:
                return 0.0  # 无平滑: 任一阶零精度 -> BLEU=0
            log_sum += w * math.log(matches / total)

        # brevity penalty(nltk 同式)
        if cand.length > ref.length:
            bp = 1.0
        else:
            bp = math.exp(1.0 - ref.length / cand.length)

        return float(bp * math.exp(log_sum))

    def similarity(self, a: str, b: str) -> float:
        """双向对称相似度(= DIVER calculate_belu_matrix 的矩阵元素)。"""
        return (self.bleu(reference=a, candidate=b) + self.bleu(reference=b, candidate=a)) / 2.0

    def diversity(self, a: str, b: str) -> float:
        """1 - similarity, 即两文本的 TD 贡献。"""
        return 1.0 - self.similarity(a, b)
