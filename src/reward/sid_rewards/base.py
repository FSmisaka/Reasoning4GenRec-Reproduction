"""SID 解析工具与 Stage 3 reward 基类。

解析语义与官方 StepRule reward(src/reward/direct_recommendation_steprule.py,
迁移自 SIDReasoner verl fork 的 direct_recommendation_StepRule_*.py)保持一致,
保证不同 reward 变体对同一 rollout 解析出完全相同的 SID。
"""

import re
from collections import defaultdict

# verl rollout worker(SIDReasoner fork 的 vllm_rollout_spmd.py)把每条 rollout
# 的 K 个 beam 候选拼成一个字符串放进 extra_info["beam_search_results"],
# 此常量为拼接分隔符, 必须与 rollout 侧保持一致。
BEAM_SEP = "<|beam_sep|>"

_SOLUTION_CLIP_CHARS = 50


def extract_sid_tokens(s: str) -> list:
    """按顺序提取 SID code token(如 ``<a_12>``)。"""
    return re.findall(r"<[^>]+>", s)


def extract_solution(solution_str: str, num_sid: int = 3, clip_chars: int = _SOLUTION_CLIP_CHARS):
    """返回 ``</think>`` 之后的前 ``num_sid`` 个 SID token。

    与原 StepRule 一致: 只看响应尾部; 无完整答案时返回 ``None``。
    """
    if len(solution_str) > clip_chars:
        solution_str = solution_str[-clip_chars:]

    match = re.search(r"</think>\s*(.*)", solution_str, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        answer_sids = extract_sid_tokens(final_answer)[:num_sid]
        if len(answer_sids) == num_sid:
            return answer_sids
    return None


def has_finished_think(solution_str: str, clip_chars: int = 200) -> bool:
    """rollout 的推理块是否以 ``</think>`` 闭合。"""
    tail = solution_str[-clip_chars:] if len(solution_str) > clip_chars else solution_str
    return "</think>" in tail


def parse_beam_candidates(beam_results, num_sid: int = 3) -> list:
    """把 ``extra_info["beam_search_results"]`` 解析为 SID 候选列表。

    rollout worker 用 ``BEAM_SEP`` 拼接 K 个候选; 每个候选至多保留前
    ``num_sid`` 个 SID token, 空片段(如推理被截断、beam 无有效 SID)丢弃。
    """
    if beam_results is None:
        return []
    if isinstance(beam_results, (list, tuple)):
        fragments = [str(x) for x in beam_results]
    else:
        fragments = str(beam_results).split(BEAM_SEP)

    candidates = []
    for fragment in fragments:
        sids = extract_sid_tokens(fragment)[:num_sid]
        if sids:
            candidates.append(sids)
    return candidates


def construct_prefix_allowed_hashmap(item_info_path: str) -> dict:
    """从 info 文件构建 {SID 前缀 -> 合法下一级 code 集合}。

    与 StepRule 实现一致, 供各 reward 变体做格式/合法性检查复用。
    """
    prefix_map = defaultdict(set)

    with open(item_info_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        semantic_id = line.split("\t")[0].strip()
        sid_list = extract_sid_tokens(semantic_id)
        if len(sid_list) != 3:
            continue

        a, b, c = sid_list
        prefix_map[(a,)].add(b)
        prefix_map[(a, b)].add(c)

    return {prefix: list(next_tokens) for prefix, next_tokens in prefix_map.items()}


def is_valid_sid_sequence(sid_list: list, prefix_map: dict) -> bool:
    """按前缀表校验 SID 三元组是否合法(不含根/终止节点)。"""
    if len(sid_list) < 3:
        return False

    a, b, c = sid_list[:3]
    if (a,) not in prefix_map:
        return False
    if b not in prefix_map[(a,)]:
        return False
    if (a, b) not in prefix_map:
        return False
    if c not in prefix_map[(a, b)]:
        return False
    return True


class BaseSidReward:
    """SID 类 GRPO reward 的统一接口。

    reward 收到解码后的 rollout(``solution_str``)、ground truth SID 字符串
    (``ground_truth``)与逐 rollout 的 ``extra_info`` dict(rollout 侧开了
    beam search 时含 ``beam_search_results``)。子类实现 :meth:`compute`,
    并容忍未知构造参数(同一份配置可被多种变体共享)。
    """

    #: 注册名, 由 @register_sid_reward 装饰器覆盖
    name = "base"

    def __init__(self, num_sid_levels: int = 3, **kwargs):
        self.num_sid_levels = int(num_sid_levels)
        # 记录并吞掉多余 kwargs, 保证共享配置可用
        self.extra_config = kwargs

    def ground_truth_sids(self, ground_truth) -> list:
        if ground_truth is None:
            return []
        return extract_sid_tokens(str(ground_truth))[: self.num_sid_levels]

    def compute(
        self,
        data_source: str,
        solution_str: str,
        ground_truth,
        extra_info: dict | None = None,
    ) -> float:
        raise NotImplementedError

    def __call__(
        self,
        data_source: str,
        solution_str: str,
        ground_truth,
        extra_info: dict | None = None,
    ) -> float:
        return self.compute(data_source, solution_str, ground_truth, extra_info)
