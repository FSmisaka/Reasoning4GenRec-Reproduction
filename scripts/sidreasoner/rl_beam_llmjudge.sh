#!/usr/bin/env bash
# SIDReasoner Stage 3 变体: 跨 Query CoT 多样性(LLM-Judge; 正确性 reward = OneRec-Think Rollout-Beam)
# 等价于 REWARD=rollout_beam INTER_DIV=1 INTER_DIV_METRIC=llmjudge bash scripts/sidreasoner/rl.sh,
# 前置: make <dataset>-sidreasoner-user-embed; judge 服务(vllm serve ../Qwen3-1.7B, 见 README);
#       Rollout-Beam 需修改版 verl fork。产物实验名自动加 _rolloutbeam_llmjudge 后缀。
set -euo pipefail
export REWARD=rollout_beam
export INTER_DIV=1
export INTER_DIV_METRIC=llmjudge
exec bash "$(dirname "$0")/rl.sh" "$@"
