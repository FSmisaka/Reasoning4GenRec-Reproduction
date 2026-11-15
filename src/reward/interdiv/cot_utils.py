"""CoT 提取与"推荐成功"判定。

SIDReasoner 的 assistant 输出格式(src/data/sidreasoner_data.py:1539)::

    <think>\\n{reasoning}\\n</think>\\n\\n<a_x><b_y><c_z>

- CoT = ``<think>`` 与 ``</think>`` 之间的文本(不含最终 SID 答案);
- "推荐成功"(多样性奖励的条件指示函数 I_{q,i})= 采样出的 SID 三元组
  与 ground truth **三级全匹配**。注意不能把原始 reward > 0 当作成功:
  StepRule 的格式分(0.1)可在零匹配时给出正奖励。
"""

import re

_THINK_OPEN_RE = re.compile(r"<think>")


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


def is_recommendation_success(
    solution_str: str, ground_truth, extract_solution_fn=None, num_sid: int = 3
) -> bool:
    """采样答案与 ground truth 三级 SID 全匹配才算推荐成功。

    Args:
        extract_solution_fn: 复用 src/reward/sid_rewards/base.py 的
            extract_solution(默认延迟导入), 保证与各 reward 变体的
            解析口径一致(只看响应尾部 50 字符)。
    """
    if extract_solution_fn is None:
        from src.reward.sid_rewards.base import extract_solution as extract_solution_fn

    answer = extract_solution_fn(solution_str, num_sid=num_sid)
    if answer is None or not ground_truth:
        return False

    from src.reward.sid_rewards.base import extract_sid_tokens

    gt = extract_sid_tokens(str(ground_truth))[:num_sid]
    return len(gt) == num_sid and answer == gt
