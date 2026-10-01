#!/usr/bin/env bash
# SIDReasoner Stage 1: SID-语言对齐 SFT(多卡 torchrun)
# GPU 选择规则:
#   1. 显式设置 CUDA_VISIBLE_DEVICES=... 时严格使用指定卡;
#   2. 否则自动挑选至多 NGPUS_WANT(默认 4)张空闲卡, 有多少用多少,
#      梯度累积会按卡数自动调整, 全局 batch_size 保持不变。
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$(bash scripts/sidreasoner/pick_gpus.sh "${NGPUS_WANT:-4}")
  export CUDA_VISIBLE_DEVICES
  echo ">>> 未指定 GPU, 自动选择空闲卡: ${CUDA_VISIBLE_DEVICES}"
fi
NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
MASTER_PORT=${MASTER_PORT:-12340}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec .venv/bin/python -m torch.distributed.run \
    --nproc_per_node "${NGPUS}" --master_port "${MASTER_PORT}" \
    -m src.train.sidreasoner_sft
