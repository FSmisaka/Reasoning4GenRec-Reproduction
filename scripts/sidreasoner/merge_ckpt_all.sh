#!/usr/bin/env bash
# SIDReasoner Stage 3: 批量合并所有 global_step_N 的 FSDP actor checkpoint
# 迁移自官方 merge_fsdp_ckpt_ALL.sh(改为调用本仓库的 merge_ckpt.py)
# 用法: CKPT_ROOT=checkpoints/RecRL_Reasoning/<Category>_stage3_rl_Qwen3-1.7B \
#       EVAL_INTERVAL=100 bash scripts/sidreasoner/merge_ckpt_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-.venv/bin/python}
: "${CKPT_ROOT:?需要设置 CKPT_ROOT, 如 checkpoints/RecRL_Reasoning/Video_Games_stage3_rl_Qwen3-1.7B}"
EVAL_INTERVAL=${EVAL_INTERVAL:-100}

if ! [[ "${EVAL_INTERVAL}" =~ ^[0-9]+$ ]] || (( EVAL_INTERVAL <= 0 )); then
    echo "EVAL_INTERVAL 必须为正整数: ${EVAL_INTERVAL}" >&2
    exit 1
fi
if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "目录不存在: ${CKPT_ROOT}" >&2
    exit 1
fi

matches=()
while IFS= read -r actor_dir; do
    step_dir="$(basename "$(dirname "${actor_dir}")")"
    if [[ "${step_dir}" =~ ^global_step_([0-9]+)$ ]]; then
        step="${BASH_REMATCH[1]}"
        if (( step % EVAL_INTERVAL == 0 )); then
            matches+=("${step}:${actor_dir}")
        fi
    fi
done < <(find "${CKPT_ROOT}" -maxdepth 2 -type d -name "actor" -path "*/global_step_*/*")

if [[ ${#matches[@]} -eq 0 ]]; then
    echo "未找到符合条件的 actor checkpoint(${CKPT_ROOT}, 每 ${EVAL_INTERVAL} 步)" >&2
    exit 1
fi

for entry in $(printf '%s\n' "${matches[@]}" | sort -t: -k1,1n); do
    step="${entry%%:*}"
    actor_dir="${entry#*:}"
    output_dir="${actor_dir}_merged"
    echo "merging step ${step}: ${actor_dir} -> ${output_dir}"
    ${PY} scripts/sidreasoner/merge_ckpt.py \
        --checkpoint "${actor_dir}" --output-dir "${output_dir}"
done
echo "all merges completed."
