"""OneReason R_div 多样性因子 x OneRec-Think Rollout-Beam 组合奖励。

实现 "OneReason Technical Report"(arXiv:2606.06260) 第 6.1 节 Reward Design
的多样性因子 Eq.(9), 并按本仓库的实验设计乘在 OneRec-Think Rollout-Beam
奖励(arXiv:2510.11639, Eq. 6-7)之上::

    R_div(CoT_{u,i}) = max(0, m_i^{(1)} - 1) / (K - 1)          (OneReason Eq.9)
    R(CoT)            = R_RolloutBeam(CoT) * R_div(CoT)         (本变体)

其中:
- K = 该条 CoT 之后并发生成的 itemic 序列(SID)条数; 在本仓库的管线里即
  rollout 侧 beam search 的宽度(actor_rollout_ref.rollout.sid_beam_size);
- m_i^{(1)} = 这 K 条 SID 中第 1 层 code(如 <a_*>)去重后的个数。论文动机:
  "the first sub-token largely determines the subsequent decoding process
  and captures coarse-grained item categories"(首层 code 粗粒度决定类别);
- R_div 在 reasoning-trace 级别定义: 同一条 CoT 的 K 条 SID 共享同一因子
  (与论文一致, 乘法组合源自论文 Eq.(7) R = R_rule * R_div)。

与论文的两点刻意差异(按实验设计):
- 论文的 R_rule 是 0/1 命中指示函数, 且 K 条 SID 为并发生成(非 beam);
  本变体保留 OneRec-Think 的 beam-max 匹配 code 数作为基础奖励,
  K 条候选来自 beam search —— 即"在 Rollout-Beam 奖励上乘 R_div"。
"""

from .base import BEAM_SEP, has_finished_think, parse_beam_candidates
from .registry import register_sid_reward
from .rollout_beam import OneRecThinkRolloutBeamReward


def count_beam_fragments(beam_results) -> int:
    """rollout 侧生成的候选条数 K。

    按原始拼接串切分计数, 包含提取失败的空片段 —— 与论文"生成 K 条序列"
    的语义一致(即使某条候选无效, 分母仍是 K)。
    """
    if beam_results is None:
        return 0
    if isinstance(beam_results, (list, tuple)):
        return len(beam_results)
    return len(str(beam_results).split(BEAM_SEP))


def r_div(beam_results, level: int = 1) -> float:
    """OneReason Eq.(9): R_div = max(0, m^{(level)} - 1) / (K - 1)。

    m^{(level)} 为 K 条候选在第 `level` 层 code(1 起始)上取值的去重个数;
    R_div ∈ [0, 1]: 首层全相同 -> 0, 全不同 -> 1; K <= 1 时无多样性可言, 返回 0。
    """
    k = count_beam_fragments(beam_results)
    if k <= 1:
        return 0.0
    idx = level - 1
    distinct = set()
    for cand in parse_beam_candidates(beam_results, num_sid=idx + 1):
        if len(cand) > idx:
            distinct.add(cand[idx])
    m = len(distinct)
    return max(0.0, m - 1) / (k - 1)


@register_sid_reward("onereason_div_rollout_beam")
class OneReasonDivRolloutBeamReward(OneRecThinkRolloutBeamReward):
    """R = R_RolloutBeam x R_div(OneReason Eq.9)。

    额外参数:
        div_level: 计算去重的 SID 层(1 = 第一层 <a_*>, 论文设定; 可试 2/3)。
    其余参数与 :class:`OneRecThinkRolloutBeamReward` 相同。
    """

    name = "onereason_div_rollout_beam"

    def __init__(self, div_level: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.div_level = int(div_level)
        if not 1 <= self.div_level <= self.num_sid_levels:
            raise ValueError(
                f"div_level 应在 [1, num_sid_levels={self.num_sid_levels}] 内, 当前 {self.div_level}"
            )

    def diversity_factor(self, beam_results) -> float:
        """该条 CoT 的多样性因子 R_div。"""
        return r_div(beam_results, level=self.div_level)

    def final_reward(self, beam_results, candidates: list, gt_sids: list, solution_str: str) -> float:
        base = self.beam_reward(candidates, gt_sids)
        reward = base * self.diversity_factor(beam_results)
        if self.format_reward_weight > 0:
            reward += self.format_reward_weight * self.format_reward(solution_str)
        return float(reward)
