#!/usr/bin/env bash
# SIDReasoner Stage 3: GRPO 强化学习(需要 SIDReasoner 官方 verl fork)
# 前置: 克隆 https://github.com/HappyPointer/SIDReasoner 到某个目录,
# 并将 config/sidreasoner.py 中 rl.verl_home 指向该目录(或用
# VERL_HOME 环境变量覆盖), 详见 README 的 SIDReasoner 小节。
set -euo pipefail
set -x
cd "$(dirname "$0")/../.."
: "${DATASET:?需要设置 DATASET=games|office|industrial}"

PY=${PY:-.venv/bin/python}
VERL_HOME=${VERL_HOME:-$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['rl']['verl_home'])")}
if [[ ! -d "${VERL_HOME}/verl" ]]; then
    echo "错误: 未找到 verl fork (${VERL_HOME}/verl)。" >&2
    echo "请克隆 SIDReasoner 官方仓库并在 config/sidreasoner.py 中设置 rl.verl_home。" >&2
    exit 1
fi

read -r NNODES TRAIN_BS MAX_PROMPT MAX_RESP LR ROLLOUT_N KL_COEF \
    EPOCHS SAVE_FREQ TEST_FREQ PROJECT USE_WANDB \
    FILTER_OVERLONG USE_REMOVE_PAD ACTOR_MB USE_KL_LOSS KL_LOSS_TYPE \
    ENTROPY_COEF GRAD_CKPT ACTOR_PARAM_OFFLOAD ACTOR_OPT_OFFLOAD \
    LOGPROB_MB TP_SIZE GPU_MEM_UTIL REF_PARAM_OFFLOAD USE_KL_REWARD \
    CRITIC_WARMUP < <(${PY} - <<'EOF'
from config import load_config
rl = load_config("sidreasoner")["rl"]
print(rl["nnodes"], rl["train_batch_size"],
      rl["max_prompt_length"], rl["max_response_length"], rl["lr"],
      rl["rollout_n"], rl["kl_loss_coef"], rl["total_epochs"],
      rl["save_freq"], rl["test_freq"], rl["project_name"],
      "True" if rl.get("use_wandb", True) else "False",
      "True" if rl.get("filter_overlong_prompts", True) else "False",
      "True" if rl.get("use_remove_padding", True) else "False",
      rl.get("actor_micro_batch_size_per_gpu", 8),
      "True" if rl.get("use_kl_loss", True) else "False",
      rl.get("kl_loss_type", "low_var_kl"),
      rl.get("entropy_coeff", 0),
      "True" if rl.get("enable_gradient_checkpointing", True) else "False",
      "True" if rl.get("actor_param_offload", False) else "False",
      "True" if rl.get("actor_optimizer_offload", False) else "False",
      rl.get("log_prob_micro_batch_size_per_gpu", 8),
      rl.get("tensor_model_parallel_size", 1),
      rl.get("gpu_memory_utilization", 0.8),
      "True" if rl.get("ref_param_offload", True) else "False",
      "True" if rl.get("use_kl_in_reward", False) else "False",
      rl.get("critic_warmup", 0))
EOF
)

# GPU 选择: 显式指定优先; 否则自动挑选至多 NGPUS_WANT 张空闲卡,
# n_gpus_per_node 以实际挑选到的卡数为准
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$(bash scripts/sidreasoner/pick_gpus.sh "${NGPUS_WANT:-4}")
  export CUDA_VISIBLE_DEVICES
  echo ">>> 未指定 GPU, 自动选择空闲卡: ${CUDA_VISIBLE_DEVICES}"
fi
N_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
if [[ "${USE_WANDB}" == "True" ]]; then
    LOGGER="['console','wandb']"
else
    LOGGER="['console']"
fi

CATEGORY=$(${PY} -c "from config import load_config; import os; print(load_config('sidreasoner')['data']['categories'][os.environ['DATASET']])")
ROOT=$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['data']['root'])")
SID_SUFFIX=$(${PY} -c "from config import load_config; from src.data.sid_switch import sid_suffix; print(sid_suffix(load_config('sidreasoner')['data']))")
DATA_DIR=$(${PY} -c "from config import load_config; rl = load_config('sidreasoner')['rl']; print(rl['data_dir'] or '')" 2>/dev/null || true)
DATA_DIR=${DATA_DIR:-${ROOT}/rec_reasoning_verl/${CATEGORY}}
STAGE2_CKPT="runs/sidreasoner/${CATEGORY}_stage2_activation${SID_SUFFIX}/final_checkpoint"

# ---- 奖励选择: REWARD 环境变量 > config rl.reward, 默认 steprule(论文原版) ----
# rollout_beam   = OneRec-Think Rollout-Beam(arXiv:2510.11639 Eq.6-7):
#   每条 rollout 的 CoT 之后由 rollout 侧做宽度 K 的 beam search 取 K 个候选
#   SID, 奖励 = 候选中匹配 ground truth 的 code 数(L 层)的最大值。
# rollout_beam_div = 在 rollout_beam 基础上乘 OneReason 多样性因子
#   R_div = max(0, m^(1)-1)/(K-1)(arXiv:2606.06260 Eq.9), m^(1) 为 K 个候选
#   第 1 层 code 去重个数。
# 两者输出目录分别加 _rolloutbeam / _rolloutbeam_div 后缀, 产物互不覆盖。
REWARD=${REWARD:-$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['rl'].get('reward', 'steprule') or 'steprule')")}
REWARD_SUFFIX=""
REWARD_FLAGS=""
case "${REWARD}" in
  steprule)
    REWARD_PATH="$(pwd)/src/reward/direct_recommendation_steprule.py"
    ;;
  rollout_beam|rollout_beam_div)
    REWARD_PATH="$(pwd)/src/reward/direct_recommendation_rolloutbeam.py"
    SID_BEAM=$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['rl'].get('sid_beam_size', 4))")
    SID_LEN=$(${PY} -c "from config import load_config; print(load_config('sidreasoner')['rl'].get('sid_length', 3))")
    if [[ "${SID_BEAM}" -le 1 ]]; then
        echo "错误: reward=${REWARD} 要求 rl.sid_beam_size > 1(当前 ${SID_BEAM})。" >&2
        echo "      否则 rollout 侧不做 beam search, 奖励将恒为 0。请改 config/sidreasoner.py。" >&2
        exit 1
    fi
    case "${REWARD}" in
      rollout_beam)
        REWARD_TYPE="onerec_rollout_beam"
        REWARD_SUFFIX="_rolloutbeam"
        ;;
      rollout_beam_div)
        REWARD_TYPE="onereason_div_rollout_beam"
        REWARD_SUFFIX="_rolloutbeam_div"
        ;;
    esac
    REWARD_FLAGS="actor_rollout_ref.rollout.sid_beam_size=${SID_BEAM} actor_rollout_ref.rollout.sid_length=${SID_LEN}"
    # reward_kwargs 在 verl 配置里是空 dict, Hydra struct 模式禁止直接新增键,
    # 必须带 '+' 前缀追加(+custom_reward_function.reward_kwargs.xxx=...)。
    # config 的 reward_kwargs.reward_type 若有值则以它为准(自定义 reward 入口),
    # 否则用 rl.reward 分支的默认实现。
    REWARD_FLAGS="${REWARD_FLAGS} $(${PY} - <<'EOF' | tr '\n' ' '
from config import load_config
rk = load_config("sidreasoner")["rl"].get("reward_kwargs", {}) or {}
for k, v in rk.items():
    v = ("true" if v else "false") if isinstance(v, bool) else v
    print(f"+custom_reward_function.reward_kwargs.{k}={v}")
EOF
)"
    if [[ "${REWARD_FLAGS}" != *"reward_kwargs.reward_type="* ]]; then
        REWARD_FLAGS="${REWARD_FLAGS} +custom_reward_function.reward_kwargs.reward_type=${REWARD_TYPE}"
    fi
    ;;
  *)
    echo "错误: 未知 rl.reward='${REWARD}', 可选: steprule | rollout_beam | rollout_beam_div" >&2
    echo "      (或直接用 scripts/sidreasoner/rl_beam*.sh / make <dataset>-sidreasoner-rl-beam[-div])" >&2
    exit 1
    ;;
esac

# ---- 跨 Query CoT 多样性奖励(DIVER 改造版): INTER_DIV=1 或 config ----
# rl.inter_query_diversity.enable=true 时开启(推荐经
# scripts/sidreasoner/rl_bleu.sh / make <dataset>-sidreasoner-rl-bleu)。
# 开启后: reward 入口 -> src/reward/direct_recommendation_interdiv.py
# (verl 自带 reward_manager=batch, 整个逻辑 batch 传入), 原 reward
# (steprule / rollout_beam 族)逐行复用不变, 实验名加 _bleu 后缀。
# 关闭时本分支零改动, Stage 3 与改造前完全一致。
INTER_DIV=${INTER_DIV:-$(${PY} -c "from config import load_config; print('True' if load_config('sidreasoner')['rl'].get('inter_query_diversity', {}).get('enable', False) else 'False')")}
case "${INTER_DIV}" in
  1|[Tt]rue|[Yy]es|[Oo]n) INTER_DIV=True ;;
  *) INTER_DIV=False ;;
esac
INTERDIV_FLAGS=""
if [[ "${INTER_DIV}" == "True" ]]; then
  read -r IQD_ENC IQD_PATH IQD_K IQD_JAC IQD_LI IQD_LF IQD_DS IQD_GAMMA \
        IQD_SIGMA IQD_DIVLEN IQD_SEED IQD_METRICS \
        < <(${PY} - <<'EOF'
from config import load_config
import os
iqd = load_config("sidreasoner")["rl"].get("inter_query_diversity", {})
cat = load_config("sidreasoner")["data"]["categories"][os.environ["DATASET"]]
enc = iqd.get("user_encoder_type", "sasrec")
path = iqd.get("user_embedding_path") or f"runs/sidreasoner/user_embed/{cat}_{enc}.npz"
print(enc, path, iqd.get("top_k_neighbors", 8), iqd.get("same_user_jaccard", 0.8),
      iqd.get("lambda_init", 0.2), iqd.get("lambda_final", 0.0),
      iqd.get("decay_steps", 300), iqd.get("gamma", 1.0),
      iqd.get("sigma", 0.5), iqd.get("bleu_div_len", 1000),
      iqd.get("seed", 0), iqd.get("metrics_path") or "")
EOF
)
  if [[ ! -f "${IQD_PATH}" ]]; then
    echo "错误: 未找到 User Embedding 缓存 ${IQD_PATH}。" >&2
    echo "      先运行: DATASET=${DATASET} make ${DATASET}-sidreasoner-user-embed" >&2
    echo "      (或修改 config/sidreasoner.py 的 rl.inter_query_diversity.user_embedding_path)" >&2
    exit 1
  fi
  # base reward 类型沿用上方 REWARD 分支的选择(steprule 默认)
  case "${REWARD}" in
    steprule)          BASE_REWARD_TYPE="steprule" ;;
    rollout_beam)      BASE_REWARD_TYPE="onerec_rollout_beam" ;;
    rollout_beam_div)  BASE_REWARD_TYPE="onereason_div_rollout_beam" ;;
  esac
  REWARD_PATH="$(pwd)/src/reward/direct_recommendation_interdiv.py"
  # batch reward manager: 把 B(query) x G(rollout) 一次性传给 reward,
  # in-batch 邻居检索在 driver 端全局 batch 上进行, 天然跨 DP rank 一致。
  INTERDIV_FLAGS="reward_model.reward_manager=batch"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.enable_inter_query_diversity=true"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.base_reward_type=${BASE_REWARD_TYPE}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.user_embedding_path=$(pwd)/${IQD_PATH}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.top_k_neighbors=${IQD_K}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.same_user_jaccard=${IQD_JAC}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.lambda_init=${IQD_LI}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.lambda_final=${IQD_LF}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.decay_steps=${IQD_DS}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.gamma=${IQD_GAMMA}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.sigma=${IQD_SIGMA}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.bleu_div_len=${IQD_DIVLEN}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.seed=${IQD_SEED}"
  INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.model_path=$(pwd)/${STAGE2_CKPT}"
  if [[ -n "${IQD_METRICS}" ]]; then
    INTERDIV_FLAGS="${INTERDIV_FLAGS} +custom_reward_function.reward_kwargs.metrics_path=$(pwd)/${IQD_METRICS}"
  fi
  REWARD_SUFFIX="${REWARD_SUFFIX}_bleu"
fi

EXPERIMENT_NAME="${CATEGORY}_stage3_rl_Qwen3-1.7B${SID_SUFFIX}${REWARD_SUFFIX}"

export PYTHONPATH="${VERL_HOME}:${PYTHONPATH:-}"
# 注意: 不要在这里设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True!
# rollout 的 vLLM sleep mode(free_cache_engine=True)依赖 CuMemAllocator 内存池,
# 与 expandable_segments 互斥, vLLM 会在引擎初始化时直接断言报错。
mkdir -p ./logs

${PY} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$(pwd)/${DATA_DIR}/train.parquet" \
    data.val_files="$(pwd)/${DATA_DIR}/test.parquet" \
    data.train_batch_size=${TRAIN_BS} \
    data.max_prompt_length=${MAX_PROMPT} \
    data.max_response_length=${MAX_RESP} \
    data.filter_overlong_prompts=${FILTER_OVERLONG} \
    data.truncation='error' \
    actor_rollout_ref.model.path="$(pwd)/${STAGE2_CKPT}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PAD} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MB} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEF} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${GRAD_CKPT} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPT_OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MB} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_MB} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD} \
    algorithm.use_kl_in_reward=${USE_KL_REWARD} \
    trainer.critic_warmup=${CRITIC_WARMUP} \
    trainer.logger="${LOGGER}" \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name="rule_base_reward" \
    ${REWARD_FLAGS} \
    ${INTERDIV_FLAGS} \
    trainer.project_name="${PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${EPOCHS} "$@" \
    2>&1 | tee "./logs/${EXPERIMENT_NAME}.log"
