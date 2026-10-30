#!/usr/bin/env bash
# SIDReasoner Stage 3 变体: OneRec-Think Rollout-Beam reward
# 等价于 REWARD=rollout_beam bash scripts/sidreasoner/rl.sh,
# 奖励参数在 config/sidreasoner.py 的 rl 段调整(见 README)。
# 前置: SIDReasoner verl fork 需含 rollout 侧 beam search 基建
# (本仓库姊妹改动, 详见 README 的 Rollout-Beam 小节)。
set -euo pipefail
export REWARD=rollout_beam
exec bash "$(dirname "$0")/rl.sh" "$@"
