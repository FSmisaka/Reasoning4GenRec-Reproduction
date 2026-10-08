#!/usr/bin/env bash
# 在独立 tmux 会话中运行 make 目标, SSH 断开后服务器上继续执行。
#
# 用法(经 Makefile):  make tmux TARGET=<目标> [SESSION=名字] [其他变量...]
#   例: make tmux TARGET=games-sidreasoner-sft SESSION=sft
#       make tmux TARGET=install-rl-env USE_MEGATRON=0 USE_SGLANG=0
#
# 常用操作:
#   接回查看   tmux attach -t <会话名>     (退出 attach 不影响任务: Ctrl-b 再按 d)
#   会话列表   tmux ls                     (断线重连后先执行它找回会话)
#   离线看日志 tail -f logs/<会话名>.log
#   结束任务   tmux kill-session -t <会话名>
set -euo pipefail
# 本脚本位于 <repo>/scripts/ 下(一层), 回到仓库根目录
cd "$(dirname "$0")/.."

# ---- 内层: 由 tmux 在服务器会话内调起, 真正执行 make ----
if [[ "${1:-}" == "--inner" ]]; then
    target=$2
    log=$3
    mkdir -p "$(dirname "${log}")" || { echo "错误: 无法创建日志目录 $(dirname "${log}")" >&2; exit 1; }
    msg="[run_in_tmux] $(date '+%F %T') 内层已启动: make ${target} (pid $$)"
    echo "${msg}"; echo "${msg}" >> "${log}"   # 重定向直写文件, 不依赖 tee
    set +e
    # tmux 面板看原始输出(含实时进度条); 落盘日志经 awk 过滤:
    # 丢弃 tqdm 的 \r 中间刷新, 仅保留每个进度条的最终状态与普通日志行
    make "${target}" 2>&1 | tee >(
        awk '{ sub(/^.*\r/, ""); print }' >> "${log}"
    )
    code=${PIPESTATUS[0]}
    sleep 1   # 等过滤进程写完, 避免退出码行与日志尾部交错
    msg="[run_in_tmux] $(date '+%F %T') make ${target} 退出码: ${code}"
    echo "${msg}"; echo "${msg}" >> "${log}"
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
for v in CUDA_VISIBLE_DEVICES CUDA_LIST NGPUS_WANT MASTER_PORT DATASET VERL_HOME \
         PY MODEL RESULT CKPT_ROOT EVAL_INTERVAL SID FORCE \
         USE_MEGATRON USE_SGLANG PIP_INDEX_URL; do
    if [[ -n "${!v:-}" ]]; then
        cmd+=" ${v}=$(printf %q "${!v}")"
    fi
done
cmd+=" bash '$(pwd)/scripts/run_in_tmux.sh' --inner '${target}' '${log}'"

tmux new-session -d -s "${session}" "${cmd}"

# 面板/命令退出后保留现场(不自动销毁会话), 便于事后 attach 查看报错
tmux set-window-option -t "${session}" remain-on-exit on 2>/dev/null || true

# 启动自检: 会话若在 2s 内消失, 说明内层命令根本没跑起来(常见原因:
# ~/.tmux.conf 在此服务器上失效, 如 macOS 的 reattach-to-user-namespace)
sleep 2
if ! tmux has-session -t "=${session}" 2>/dev/null; then
    echo "错误: 会话 ${session} 启动后立即退出, 内层命令未执行(日志 ${log} 也未生成)。" >&2
    echo "排查步骤:" >&2
    echo "  1) tmux new -d -s t1; sleep 2; tmux ls   # 空会话也秒退 -> tmux 环境问题" >&2
    echo "  2) tmux -f /dev/null new -d -s t2; sleep 2; tmux ls   # 绕过 ~/.tmux.conf 后正常 -> 修配置" >&2
    echo "  3) bash '$(pwd)/scripts/run_in_tmux.sh' --inner ${target} ${log}   # 前台直跑看真实报错" >&2
    [[ -f "${log}" ]] && { echo "--- 日志尾部 ---"; tail -20 "${log}"; }
    exit 1
fi

echo ">>> 已在 tmux 会话 '${session}' 后台启动: make ${target}"
echo "    本终端到此为止, 不会再有训练输出(任务在后台会话中, 断开 SSH 也继续)"
echo "    接回查看: tmux attach -t ${session}   (退出 attach: Ctrl-b 再按 d, 不停任务)"
echo "    离线日志: tail -f ${log}"
