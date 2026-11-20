"""LLM-as-a-Judge CoT 相似度 scorer(vLLM OpenAI 兼容接口, 异步批量)。

judge_prompt.txt 让 judge 对两条 CoT 按 evidence / strategy / profile 三个
维度评 0-4 相似度, sim = mean(score) / 4, diversity = 1 - sim。

块(block)协议: judge 需要用户历史, 而 DiverBleu/NgramJaccard 的
``similarity(a, b)`` 只收文本。约定 a/b 为
``render_block(history_lines, cot)`` 渲染的块, 内部按 ``<<COT>>`` 行
切开历史与 CoT 再填模板。

judge 服务(需提前启动, 见 README):
    vllm serve ../Qwen3-1.7B --served-model-name Qwen3-1.7B --port 8000
"""

import asyncio
import json
import os

_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judge_prompt.txt")
BLOCK_SEP = "\n<<COT>>\n"
_KEYS = ("evidence", "strategy", "profile")


def render_block(history_lines: list, cot: str) -> str:
    return "\n".join(history_lines) + BLOCK_SEP + cot


def split_block(block: str) -> tuple:
    history, _, cot = block.partition(BLOCK_SEP)
    return history, cot


def parse_scores(text: str):
    """从 judge 输出解析三个维度分(嵌套 score 或扁平整数); 失败返回 None。"""
    text = text or ""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        out = {}
        for k in _KEYS:
            v = obj[k]
            v = v["score"] if isinstance(v, dict) else v
            out[k] = min(4, max(0, int(v)))
        return out
    except (KeyError, TypeError, ValueError):
        return None


class HistoryLookup:
    """(split, index) -> ['<a_x><b_y><c_z> - title', ...], 源自原始 split CSV。

    CSV 与训练 parquet 行号一一一致(同 src/data/sidreasoner_user_embed.py
    的对齐口径), history_item_sid / history_item_title 列为 list 的 str repr。
    data_root 为 split 目录的上级(默认 data/raw/Amazon)。
    """

    def __init__(self, data_root: str, category: str):
        self.data_root = data_root
        self.category = category
        self._cache: dict = {}

    def _csv(self, split: str):
        import pandas as pd

        if split not in self._cache:
            path = os.path.join(
                self.data_root, split, f"{self.category}_5_2016-10-2018-11.csv"
            )
            self._cache[split] = pd.read_csv(
                path, usecols=["history_item_sid", "history_item_title"]
            )
        return self._cache[split]

    def lines(self, split: str, index: int) -> list:
        import ast

        row = self._csv(str(split)).iloc[int(index)]
        sids = ast.literal_eval(row["history_item_sid"])
        titles = ast.literal_eval(row["history_item_title"])
        return [f"{s} - {t}" for s, t in zip(sids, titles)]


class LLMJudge:
    """异步批量 CoT 相似度 judge。

    Args:
        base_url: OpenAI 兼容服务地址(vLLM serve)。
        model: served model 名。
        concurrency: 并发请求数上限。
        temperature / max_tokens: judge 解码参数(输出是一小段 JSON)。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen3-1.7B",
        prompt_path: str = _PROMPT_PATH,
        concurrency: int = 32,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.base_url = base_url
        self.model = model
        self.concurrency = int(concurrency)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        with open(prompt_path) as f:
            self.template = f.read()
        self.n_calls = 0
        self.n_fallback = 0
        # 最近一次 judge_pairs 的逐对原始输出/是否解析成功(与输入对一一对应)
        self.last_raw = []
        self.last_parsed = []

    async def _judge_async(self, pairs: list) -> list:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self.base_url, api_key="EMPTY")
        sem = asyncio.Semaphore(self.concurrency)

        async def one(a: str, b: str):
            ha, ca = split_block(a)
            hb, cb = split_block(b)
            prompt = (
                self.template.replace("{{history_a}}", ha)
                .replace("{{cot_a}}", ca)
                .replace("{{history_b}}", hb)
                .replace("{{cot_b}}", cb)
            )
            async with sem:
                r = await client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            self.n_calls += 1
            content = r.choices[0].message.content or ""
            scores = parse_scores(content)
            if scores is None:
                self.n_fallback += 1
                return {k: 2 for k in _KEYS}, False, content
            return scores, True, content

        results = await asyncio.gather(*(one(a, b) for a, b in pairs))
        await client.close()
        self.last_raw = [raw for _, _, raw in results]
        self.last_parsed = [ok for _, ok, _ in results]
        return [scores for scores, _, _ in results]

    def judge_pairs(self, pairs: list) -> list:
        """逐对返回 {dim: 0..4}; 解析失败回退全 2(中性相似度 0.5)。"""
        return asyncio.run(self._judge_async(list(pairs)))

    def similarities(self, pairs: list) -> list:
        scores = self.judge_pairs(pairs)
        return [sum(s.values()) / (4.0 * len(_KEYS)) for s in scores]

    def similarity(self, a: str, b: str) -> float:
        return self.similarities([(a, b)])[0]

    def diversity(self, a: str, b: str) -> float:
        return 1.0 - self.similarity(a, b)

    def diversities(self, target: str, refs: list) -> list:
        """一条 rollout vs K 条 reference, 一次批量请求。"""
        return [1.0 - s for s in self.similarities([(target, r) for r in refs])]
