#!/usr/bin/env bash
# SIDReasoner Stage 2(GEM): 推理激活 SFT 的 GEM 损失变体(多卡 torchrun,
# 默认输入为 GEM Stage 1 的 final_checkpoint)
# GPU 选择规则同 sft_gem.sh; 默认 NGPUS_WANT=2(适配 2x A800-80G 服务器)。
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$(bash scripts/sidreasoner/pick_gpus.sh "${NGPUS_WANT:-2}")
  export CUDA_VISIBLE_DEVICES
  echo ">>> 未指定 GPU, 自动选择空闲卡: ${CUDA_VISIBLE_DEVICES}"
fi
NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
MASTER_PORT=${MASTER_PORT:-29520}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec .venv/bin/python -m torch.distributed.run \
    --nproc_per_node "${NGPUS}" --master_port "${MASTER_PORT}" \
    -m src.train.sidreasoner_activation_gem
