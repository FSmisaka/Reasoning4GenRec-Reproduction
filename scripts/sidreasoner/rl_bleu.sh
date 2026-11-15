#!/usr/bin/env bash
# SIDReasoner Stage 3 变体: 跨 Query CoT 多样性奖励(DIVER 改造版)
# 等价于 INTER_DIV=1 bash scripts/sidreasoner/rl.sh,
# 在原 reward(steprule, 可用 REWARD= 换成 beam 族)上叠加
# R' = R + lambda*I*clip(gamma^T * TD, 0, sigma), 参数在
# config/sidreasoner.py 的 rl.inter_query_diversity 段调整。
# 前置: make <dataset>-sidreasoner-user-embed 生成离线 User Embedding。
set -euo pipefail
export INTER_DIV=1
exec bash "$(dirname "$0")/rl.sh" "$@"
