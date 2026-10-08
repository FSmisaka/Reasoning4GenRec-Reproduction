#!/usr/bin/env bash
# 在独立 tmux 会话中运行 make 目标, SSH 断开后服务器上继续执行。
#
# 用法(经 Makefile):  make tmux TARGET=games-sidreasoner-sft [SESSION=名字]
# 直接调用:           bash scripts/run_in_tmux.sh <make-target> [会话名]
#
# 常用操作:
#   接回查看   tmux attach -t <会话名>     (退出 attach 不影响任务: Ctrl-b 再按 d)
#   会话列表   tmux ls
#   离线看日志 tail -f logs/<会话名>.log
#   结束任务   tmux kill-session -t <会话名>
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- 内层: 由 tmux 在服务器会话内调起, 真正执行 make ----
if [[ "${1:-}" == "--inner" ]]; then
    target=$2
    log=$3
    mkdir -p "$(dirname "${log}")"
    set +e
    make "${target}" 2>&1 | tee -a "${log}"
    code=${PIPESTATUS[0]}
    echo "[run_in_tmux] make ${target} 退出码: ${code}" | tee -a "${log}"
    exit "${code}"
fi

# ---- 外层: 创建 tmux 会话 ----
target=${1:?用法: bash scripts/run_in_tmux.sh <make-target> [会话名]}
session=${2:-r4g_$(echo "${target}" | tr -c 'a-zA-Z0-9' '_')}
log="logs/${session}.log"

command -v tmux >/dev/null 2>&1 || {
    echo "错误: 未安装 tmux (Ubuntu: sudo apt install tmux; CentOS: sudo yum install tmux)" >&2
    exit 1
}
if tmux has-session -t "=${session}" 2>/dev/null; then
    echo "错误: 会话 ${session} 已存在 (tmux attach -t ${session} 查看后 Ctrl-d 或换会话名)" >&2
    exit 1
fi

# tmux 服务端不继承当前客户端的命令行环境变量,
# 这里显式转发本项目相关的变量到内层 make
cmd="env"
for v in CUDA_VISIBLE_DEVICES NGPUS_WANT MASTER_PORT DATASET VERL_HOME \
         PY MODEL RESULT CKPT_ROOT EVAL_INTERVAL; do
    if [[ -n "${!v:-}" ]]; then
        cmd+=" ${v}=$(printf %q "${!v}")"
    fi
done
cmd+=" bash '$(pwd)/scripts/run_in_tmux.sh' --inner '${target}' '${log}'"

tmux new-session -d -s "${session}" "${cmd}"
echo ">>> 已在 tmux 会话 '${session}' 后台启动: make ${target}"
echo "    接回查看: tmux attach -t ${session}   (退出 attach: Ctrl-b 再按 d)"
echo "    离线日志: tail -f ${log}"
