#!/usr/bin/env bash
# SIDReasoner Stage 3 变体: Niche(小众物品)多样性(正确性 reward = OneRec-Think Rollout-Beam)
# 等价于 REWARD=rollout_beam INTER_DIV=1 INTER_DIV_METRIC=niche bash scripts/sidreasoner/rl.sh,
# 塑形/门控参数取 config 的 rl.inter_query_diversity。
# 产物实验名自动加 _rolloutbeam_niche 后缀。
# 前置: make <dataset>-sidreasoner-user-embed && make <dataset>-sidreasoner-niche-data;
#       Rollout-Beam 需修改版 verl fork。
set -euo pipefail
export REWARD=rollout_beam
export INTER_DIV=1
export INTER_DIV_METRIC=niche
exec bash "$(dirname "$0")/rl.sh" "$@"
