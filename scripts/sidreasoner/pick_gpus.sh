#!/usr/bin/env bash
# 从 nvidia-smi 挑选至多 $1 张空闲 GPU, 输出逗号分隔的编号。
# 空闲标准与 Makefile 一致: 显存占用 < 1000MiB 且利用率 < 10%,
# 按剩余显存从多到少排序取前 N 张(不足 N 张时有多少给多少)。
set -euo pipefail

WANT=${1:-4}
PICKS=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '$2 < 1000 && $3 < 10 {print $2, $1}' \
    | sort -n | head -n "${WANT}" | awk '{print $2}' | paste -sd, -)

if [[ -z "${PICKS}" ]]; then
    echo "错误: 未找到空闲 GPU(标准: 显存<1000MiB 且利用率<10%)。" >&2
    echo "可稍后重试, 或手动指定: CUDA_VISIBLE_DEVICES=N[,M,...]" >&2
    exit 1
fi
echo "${PICKS}"
