"""CoT 提取与"推荐成功"判定。

SIDReasoner 的 assistant 输出格式(src/data/sidreasoner_data.py:1539)::

    <think>\\n{reasoning}\\n</think>\\n\\n<a_x><b_y><c_z>

- CoT = ``<think>`` 与 ``</think>`` 之间的文本(不含最终 SID 答案);
- "推荐成功"指示函数 I_{q,i} 支持三档口径(success_mode):
    - ``full``(默认): SID 三元组与 ground truth **三级全匹配**;
    - ``prefix``: 至少第 1 层 code 匹配(前缀连续匹配 >=1 层), 大幅提高
      低成功率阶段的覆盖率;
    - ``none``: 不做推荐成功约束(所有 rollout 参与多样性奖励)。
  注意不能把原始 reward > 0 当作成功: StepRule 的格式分(0.1)可在零匹配
  时给出正奖励。
"""

import re

_THINK_OPEN_RE = re.compile(r"<think>")

SUCCESS_MODES = ("full", "prefix", "none")


def extract_cot(response_str: str) -> str | None:
    """提取 ``<think> ... </think>`` 之间的推理文本。

    - ``</think>`` 缺失(rollout 被截断)返回 None;
    - ``<think>`` 缺失(Qwen3 chat template 由服务端注入, 解码文本可能
      不含)时取 ``</think>`` 之前的全部文本。
    """
    if response_str is None:
        return None
    close = response_str.rfind("</think>")
    if close == -1:
        return None
    m = _THINK_OPEN_RE.search(response_str)
    start = m.end() if m is not None else 0
    return response_str[start:close].strip()


def recommendation_match(
    solution_str: str, ground_truth, extract_solution_fn=None, num_sid: int = 3
):
    """一次解析返回 (answer_parsed, prefix_len)。

    answer_parsed: 能否从响应尾部解析出完整 SID 三元组;
    prefix_len: 与 ground truth 沿层级的连续前缀匹配长度(0..num_sid)。
    """
    if extract_solution_fn is None:
        from src.reward.sid_rewards.base import extract_solution as extract_solution_fn

    answer = extract_solution_fn(solution_str, num_sid=num_sid)
    if answer is None:
        return False, 0
    from src.reward.sid_rewards.base import extract_sid_tokens

    gt = extract_sid_tokens(str(ground_truth or ""))[:num_sid]
    if len(gt) != num_sid:
        return True, 0
    matched = 0
    for pred, gold in zip(answer, gt):
        if pred != gold:
            break
        matched += 1
    return True, matched


def recommendation_prefix_len(
    solution_str: str, ground_truth, extract_solution_fn=None, num_sid: int = 3
) -> int:
    """采样答案与 ground truth 沿 SID 层级的连续前缀匹配长度(0..num_sid)。"""
    _, matched = recommendation_match(
        solution_str, ground_truth, extract_solution_fn=extract_solution_fn, num_sid=num_sid
    )
    return matched


def is_recommendation_success(
    solution_str: str,
    ground_truth,
    extract_solution_fn=None,
    num_sid: int = 3,
    mode: str = "full",
) -> bool:
    """推荐成功指示 I(mode: full | prefix | none)。

    Args:
        extract_solution_fn: 复用 src/reward/sid_rewards/base.py 的
            extract_solution(默认延迟导入), 保证与各 reward 变体的
            解析口径一致(只看响应尾部 50 字符)。
        mode: "full"=三级全匹配(默认, 严格); "prefix"=第 1 层匹配即可;
            "none"=恒为 True(仅要求答案可解析, 见下)。
    """
    if mode not in SUCCESS_MODES:
        raise ValueError(f"未知 success mode: {mode!r}, 可选 {SUCCESS_MODES}")
    parsed, matched = recommendation_match(
        solution_str, ground_truth, extract_solution_fn=extract_solution_fn, num_sid=num_sid
    )
    if not parsed:
        return False
    if mode == "none":
        # 不做正确性约束, 但仍要求答案可解析(格式坏/截断的 rollout 不参与)
        return True
    return matched >= (num_sid if mode == "full" else 1)
