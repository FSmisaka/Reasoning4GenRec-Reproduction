#!/usr/bin/env bash
# SIDReasoner Stage 3 变体: 跨 Query CoT 多样性(n-gram 集合 Jaccard 版)
# 等价于 INTER_DIV=1 INTER_DIV_METRIC=ngram bash scripts/sidreasoner/rl.sh,
# 多样性度量 = 1 - |G_n(CoT1)∩G_n(CoT2)| / |G_n(CoT1)∪G_n(CoT2)|,
# 产物实验名自动加 _ngram 后缀(与 _bleu / _rolloutbeam* 互不覆盖)。
# 前置: make <dataset>-sidreasoner-user-embed 生成离线 User Embedding。
set -euo pipefail
export INTER_DIV=1
export INTER_DIV_METRIC=ngram
exec bash "$(dirname "$0")/rl.sh" "$@"
