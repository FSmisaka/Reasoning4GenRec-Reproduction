#!/usr/bin/env bash
# SIDReasoner 跨 Query 多样性奖励: User Embedding 离线提取
# 用法: DATASET=games bash scripts/sidreasoner/user_embed.sh
#   可选: ENCODER=bge_mean / OUT=... / BATCH_SIZE=... / DEVICE=cuda
# 默认参数取 config/sidreasoner.py 的 rl.inter_query_diversity。
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"

PY=${PY:-.venv/bin/python}

ARGS=()
[[ -z "${ENCODER:-}" ]]     || ARGS+=(--encoder "${ENCODER}")
[[ -z "${OUT:-}" ]]         || ARGS+=(--out "${OUT}")
[[ -z "${BATCH_SIZE:-}" ]]  || ARGS+=(--batch-size "${BATCH_SIZE}")
[[ -z "${DEVICE:-}" ]]      || ARGS+=(--device "${DEVICE}")
[[ -z "${SASREC_CKPT:-}" ]] || ARGS+=(--sasrec-checkpoint "${SASREC_CKPT}")
[[ -z "${BGE_EMB:-}" ]]     || ARGS+=(--bge-item-embeddings "${BGE_EMB}")

DATASET=${DATASET} ${PY} -m src.data.sidreasoner_user_embed "${ARGS[@]}"
