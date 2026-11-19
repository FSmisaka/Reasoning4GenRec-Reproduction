#!/usr/bin/env bash
# SIDReasoner Stage 3 变体: 跨 Query CoT 多样性(n-gram Jaccard; 正确性 reward = Rollout-Beam)
# 等价于 REWARD=rollout_beam INTER_DIV=1 INTER_DIV_METRIC=ngram bash scripts/sidreasoner/rl.sh,
# 合并方式(combine=add|mul)与门控参数取 config 的 rl.inter_query_diversity。
# 产物实验名自动加 _rolloutbeam_ngram 后缀。
# 前置: make <dataset>-sidreasoner-user-embed; Rollout-Beam 需修改版 verl fork。
set -euo pipefail
export REWARD=rollout_beam
export INTER_DIV=1
export INTER_DIV_METRIC=ngram
exec bash "$(dirname "$0")/rl.sh" "$@"
