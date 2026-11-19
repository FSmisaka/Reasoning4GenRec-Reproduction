#!/usr/bin/env bash
# SIDReasoner Stage 3 通用入口: 正确性 reward x 多样性 reward x 合并方式
# 用法: CORR=<...> DIV=<...> [COMBINE=<...>] bash scripts/sidreasoner/rl_div.sh
#   CORR    steprule(默认) | beam | beam-div
#   DIV     bleu(默认) | ngram | none(关闭多样性, 等价普通 rl.sh)
#   COMBINE add(默认) | mul   (覆盖 config 的 rl.inter_query_diversity.combine)
# 环境变量均经 make / make tmux 透传(见 scripts/run_in_tmux.sh)。
set -euo pipefail

case "${CORR:-steprule}" in
  steprule)  export REWARD=steprule ;;
  beam)      export REWARD=rollout_beam ;;
  beam-div)  export REWARD=rollout_beam_div ;;
  *) echo "错误: 未知 CORR='${CORR}' (可选 steprule | beam | beam-div)" >&2; exit 1 ;;
esac

case "${DIV:-bleu}" in
  bleu)  export INTER_DIV=1; export INTER_DIV_METRIC=bleu ;;
  ngram) export INTER_DIV=1; export INTER_DIV_METRIC=ngram ;;
  none)  unset INTER_DIV INTER_DIV_METRIC || true ;;
  *) echo "错误: 未知 DIV='${DIV}' (可选 bleu | ngram | none)" >&2; exit 1 ;;
esac

# COMBINE 直接透传(rl.sh 内校验并覆盖 config)
exec bash "$(dirname "$0")/rl.sh" "$@"
