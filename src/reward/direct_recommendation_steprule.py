"""
SIDReasoner Stage 3(RL)的 stepwise 规则奖励

迁移自 SIDReasoner 官方仓库 verl/utils/reward_score/
direct_recommendation_StepRule_{Games,Industrial,Office}.py
(三者仅 info 文件路径不同, 此处按 DATASET 环境变量统一解析)。

奖励构成(对应论文 3.3.2 节):
- R_sr 步进规则奖励: SID 三级前缀逐级匹配, 0.25 -> 0.5 -> 1.0
- R_f 格式奖励(权重 0.1): 生成的 SID 序列在 info 前缀树中合法

由 verl 以 custom_reward_function.path 方式加载,
入口函数为 rule_base_reward。
"""

import os
import re
from collections import defaultdict

_SOLUTION_CLIP_CHARS = 50

_CATEGORY_FILES = {
    "Video_Games": "Video_Games_5_2016-10-2018-11.txt",
    "Office_Products": "Office_Products_5_2016-10-2018-11.txt",
    "Industrial_and_Scientific": (
        "Industrial_and_Scientific_5_2016-10-2018-11.txt"
    ),
}


def _resolve_info_path():
    import sys

    sys.path.insert(0, ".")
    from config import load_config

    cfg = load_config("sidreasoner")
    category = os.environ.get("DATASET")
    if category is None or category not in cfg["data"]["categories"]:
        raise RuntimeError(
            f"DATASET 环境变量未设置或无效: {category!r}, "
            f"可选: {sorted(cfg['data']['categories'])}"
        )
    fname = _CATEGORY_FILES.get(cfg["data"]["categories"][category])
    return os.path.join(cfg["data"]["root"], "info", fname)


def extract_sid_tokens(s: str) -> list:
    pattern = r"<[^>]+>"
    return re.findall(pattern, s)


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    match = re.search(r"</think>\s*(.*)", solution_str, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        answer_sids = extract_sid_tokens(final_answer)[:3]
        if len(answer_sids) == 3:
            return answer_sids
    return None


def calculate_reward(answer_sids, ground_truth_sids):
    current_score = 0.0
    if answer_sids[0] == ground_truth_sids[0]:
        current_score += 0.25
        if answer_sids[1] == ground_truth_sids[1]:
            current_score *= 2
            if answer_sids[2] == ground_truth_sids[2]:
                current_score *= 2
    return current_score


def calculate_format_reward(answer_sids, prefix_map):
    def is_valid_sid_sequence(sid_list):
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

    return is_valid_sid_sequence(answer_sids)


def construct_prefix_allowed_hashmap(item_info_path):
    sid_pattern = re.compile(r"<[^>]+>")
    prefix_map = defaultdict(set)
    with open(item_info_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        semantic_id = line.split("\t")[0].strip()
        sid_list = sid_pattern.findall(semantic_id)
        if len(sid_list) != 3:
            continue
        a, b, c = sid_list
        prefix_map[(a,)].add(b)
        prefix_map[(a, b)].add(c)
    return {
        prefix: list(next_tokens)
        for prefix, next_tokens in prefix_map.items()
    }


class MyRewardComputer:
    def __init__(self):
        self.sid_hash = construct_prefix_allowed_hashmap(
            _resolve_info_path()
        )

    def compute(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: dict | None = None,
    ) -> float:
        answer = extract_solution(solution_str=solution_str)
        ground_truth = extract_sid_tokens(ground_truth)[:3]
        if answer is None:
            return 0
        return calculate_reward(
            answer, ground_truth
        ) + 0.1 * calculate_format_reward(answer, self.sid_hash)


_reward_computer = None


def _get_reward_computer():
    global _reward_computer
    if _reward_computer is None:
        _reward_computer = MyRewardComputer()
    return _reward_computer


def rule_base_reward(data_source, solution_str, ground_truth, extra_info=None):
    rc = _get_reward_computer()
    return rc.compute(data_source, solution_str, ground_truth, extra_info)
