"""OneRec-Think Rollout-Beam reward(SIDReasoner Stage 3 GRPO 变体)。

实现 "OneRec-Think: In-Text Reasoning for Generative Recommendation"
(arXiv:2510.11639) 第 4.3 节 Eq. (6)-(7)::

    R_Rollout-Beam = max_{s_hat in B(tau)} sum_{l=1..L} 1[s_hat^l == s_gt^l]

其中 ``tau`` 是某条 rollout 的思维链 CoT, ``B(tau)`` 是在该 CoT 之后、
以宽度 K 对 ``P(s | prompt, tau; theta)`` 做 beam search 得到的 top-K SID
补全。换言之: 一条 rollout 的奖励取决于"模型沿这条思维链最多能推荐到
多准", 而不是它恰好采样出的那一个 SID。

beam search 由 SIDReasoner verl fork 的 vLLM rollout worker 执行
(rl.sh 里 actor_rollout_ref.rollout.sid_beam_size=K>1 + sid_length=L 开启),
K 个候选经 extra_info["beam_search_results"] 传入本函数。

与默认 StepRule reward(src/reward/direct_recommendation_steprule.py,
对采样 SID 给 0.25/0.5/1.0 步进分)的差异:

- 评分对象从"采样出的单个 SID"变为"K 个 beam 候选中最好的一个";
- 分值 = 匹配的 SID code 数(0..L), 不再是逐级翻倍;
- 论文未定义格式分, 默认关闭(reward_kwargs.format_reward_weight > 0 可开)。
"""

from .base import (
    BaseSidReward,
    construct_prefix_allowed_hashmap,
    extract_solution,
    has_finished_think,
    is_valid_sid_sequence,
    parse_beam_candidates,
)
from .registry import register_sid_reward


@register_sid_reward("onerec_rollout_beam")
class OneRecThinkRolloutBeamReward(BaseSidReward):
    """Rollout-Beam reward(OneRec-Think, Eq. 6-7)。

    Args:
        num_sid_levels: SID 层数 L(``<a_*><b_*><c_*>`` 为 3)。
        match_mode:
            - ``"position"``(默认, 忠实论文): 各层独立比对,
              ``sum_l 1[s_hat^l == s_gt^l]``;
            - ``"prefix"``: 从第一层起连续匹配才计数(前面前缀不匹配则后面
              不计), 与 StepRule 的层级思想一致, 供对照实验。
        normalize: 匹配 code 数除以 L, 奖励落到 [0, 1]。默认 False(论文为
            未归一的匹配计数)。
        format_reward_weight: 对 rollout 自己采样出的合法 SID 三元组的附加
            格式分权重。默认 0.0(论文无格式项)。
        item_info_path: 格式检查用的 info 文件路径(format_reward_weight>0 时
            必需)。
        require_beam_results: 缺少 beam 结果时直接报错并给出修复提示(默认
            True, 防止"忘了开 beam search 却全零奖励"的静默错误)。
        no_think_reward: CoT 未闭合(``</think>`` 缺失)时的奖励, 默认 0.0。
    """

    name = "onerec_rollout_beam"

    def __init__(
        self,
        num_sid_levels: int = 3,
        match_mode: str = "position",
        normalize: bool = False,
        format_reward_weight: float = 0.0,
        item_info_path: str = None,
        require_beam_results: bool = True,
        no_think_reward: float = 0.0,
        **kwargs,
    ):
        super().__init__(num_sid_levels=num_sid_levels, **kwargs)
        assert match_mode in ("position", "prefix"), f"未知 match_mode: {match_mode}"
        self.match_mode = match_mode
        self.normalize = bool(normalize)
        self.format_reward_weight = float(format_reward_weight)
        self.require_beam_results = bool(require_beam_results)
        self.no_think_reward = float(no_think_reward)
        self.prefix_map = None
        if self.format_reward_weight > 0:
            if not item_info_path:
                raise ValueError(
                    "format_reward_weight > 0 需要 item_info_path"
                    "(如 data/raw/Amazon/info/Video_Games_5_2016-10-2018-11.txt)。"
                )
            self.prefix_map = construct_prefix_allowed_hashmap(item_info_path)

    # ------------------------------------------------------------------ #
    # 核心打分
    # ------------------------------------------------------------------ #

    def matched_codes(self, candidate: list, gt_sids: list) -> int:
        """单个 beam 候选与 ground truth 的匹配 code 数。"""
        if self.match_mode == "prefix":
            matched = 0
            for pred, gold in zip(candidate, gt_sids):
                if pred != gold:
                    break
                matched += 1
            return matched

        return sum(1 for pred, gold in zip(candidate, gt_sids) if pred == gold)

    def beam_reward(self, candidates: list, gt_sids: list) -> float:
        """R_Rollout-Beam = beam 候选匹配数的最大值。"""
        best = max((self.matched_codes(c, gt_sids) for c in candidates), default=0)
        if self.normalize and self.num_sid_levels > 0:
            return best / self.num_sid_levels
        return float(best)

    def format_reward(self, solution_str: str) -> float:
        """可选: rollout 自己采样的 SID 三元组结构合法性的附加分。"""
        if self.format_reward_weight <= 0 or self.prefix_map is None:
            return 0.0
        answer = extract_solution(solution_str, num_sid=self.num_sid_levels)
        if answer is None:
            return 0.0
        return float(is_valid_sid_sequence(answer, self.prefix_map))

    # ------------------------------------------------------------------ #
    # verl 入口
    # ------------------------------------------------------------------ #

    def compute(
        self,
        data_source: str,
        solution_str: str,
        ground_truth,
        extra_info: dict | None = None,
    ) -> float:
        gt_sids = self.ground_truth_sids(ground_truth)
        if len(gt_sids) < self.num_sid_levels:
            return 0.0

        beam_results = (extra_info or {}).get("beam_search_results", None)
        if beam_results is None:
            if self.require_beam_results:
                raise RuntimeError(
                    "OneRecThinkRolloutBeamReward 需要逐 rollout 的 beam search 结果"
                    "(extra_info['beam_search_results'])但没有找到。请在 rl.sh 的"
                    "训练命令里开启 actor_rollout_ref.rollout.sid_beam_size=K(>1) 与"
                    " actor_rollout_ref.rollout.sid_length=L, 或把"
                    " rl.reward_kwargs.require_beam_results 设为 False 回退为零奖励。"
                )
            return 0.0

        # CoT 未闭合的 rollout, beam 无从落在 SID 上
        if not has_finished_think(solution_str):
            return self.no_think_reward

        candidates = parse_beam_candidates(beam_results, num_sid=self.num_sid_levels)
        reward = self.beam_reward(candidates, gt_sids)
        if self.format_reward_weight > 0:
            reward += self.format_reward_weight * self.format_reward(solution_str)
        return float(reward)
