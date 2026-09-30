#!/usr/bin/env bash
# SIDReasoner 评估(非思考模式): 拆分测试集 -> 多卡并行 trie 约束束搜 -> 合并 -> 计算指标
# 待评模型由 config/sidreasoner.py 的 evaluation.eval_model 决定:
#   "stage1" | "stage2" | checkpoint 完整路径
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"

PY=.venv/bin/python
if [[ -n "${CUDA_LIST:-}" ]]; then
  CUDA_LIST_CSV=${CUDA_LIST// /,}
else
  CUDA_LIST=${CUDA_VISIBLE_DEVICES:-0 1}
  CUDA_LIST=${CUDA_LIST//,/ }
  CUDA_LIST_CSV=${CUDA_LIST// /,}
fi

CATEGORY=$(${PY} -c "from config import load_config; import os; print(load_config('sidreasoner')['data']['categories'][os.environ['DATASET']])")
ROOT=$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['data']['root'])")
TEST_FILE="${ROOT}/test/${CATEGORY}_5_2016-10-2018-11.csv"
INFO_FILE="${ROOT}/info/${CATEGORY}_5_2016-10-2018-11.txt"

EVAL_MODEL=${MODEL:-$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['evaluation']['eval_model'])")}
EVAL_LABEL=$(basename "${EVAL_MODEL}")
TEMP_DIR="./temp/sidreasoner/${CATEGORY}-${EVAL_LABEL}"
RESULT_DIR="./results/sidreasoner"
mkdir -p "${TEMP_DIR}" "${RESULT_DIR}"

echo "Evaluating ${CATEGORY} (model=${EVAL_MODEL}, GPUs=${CUDA_LIST_CSV})"
${PY} -m src.eval.sidreasoner_metrics split \
    --input_path "${TEST_FILE}" --output_path "${TEMP_DIR}" \
    --cuda_list "${CUDA_LIST_CSV}"

for i in ${CUDA_LIST}; do
    if [[ -f "${TEMP_DIR}/${i}.csv" ]]; then
        echo "Starting evaluation on GPU ${i}"
        TEST_SHARD="${TEMP_DIR}/${i}.csv" RESULT_SHARD="${TEMP_DIR}/${i}.json" \
            CUDA_VISIBLE_DEVICES=${i} ${PY} -c "
import os
from src.eval.sidreasoner_eval import main
main(test_data_path=os.environ['TEST_SHARD'],
     result_json_data=os.environ['RESULT_SHARD'])
" &
    else
        echo "Warning: split file ${TEMP_DIR}/${i}.csv not found, skipping GPU ${i}"
    fi
done
wait

ACTUAL=""
for gpu in ${CUDA_LIST}; do
    [[ -f "${TEMP_DIR}/${gpu}.json" ]] && ACTUAL="${ACTUAL}${gpu},"
done
ACTUAL=${ACTUAL%,}
[[ -n "${ACTUAL}" ]] || { echo "Error: no result shards"; exit 1; }

FINAL="${RESULT_DIR}/final_result_${CATEGORY}.json"
${PY} -m src.eval.sidreasoner_metrics merge \
    --input_path "${TEMP_DIR}" --output_path "${FINAL}" \
    --cuda_list "${ACTUAL}"

${PY} -m src.eval.sidreasoner_metrics calc --path "${FINAL}" --item_path "${INFO_FILE}"
echo "Results saved to ${FINAL}"
