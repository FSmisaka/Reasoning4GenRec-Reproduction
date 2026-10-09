#!/bin/bash
# 配置说明:
#   PIP_INDEX_URL  PyPI 镜像, 默认清华源(只加速 PyPI 包; GitHub
#                  release 的 wheel 与 git+https 安装不走镜像,
#                  置空则恢复官方源)
#   PIP / PY       pip 与 python 解释器, 默认 venv 内; docker 内
#                  可 PIP=pip3 PY=python3 覆盖
cd "$(dirname "$0")/../.."

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
PIP=${PIP:-.venv/bin/pip}
PY=${PY:-.venv/bin/python}

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
if [ $USE_SGLANG -eq 1 ]; then
    "${PIP}" install "sglang[all]==0.4.6.post1" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python && "${PIP}" install torch-memory-saver --no-cache-dir
fi
"${PIP}" install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata

echo "2. install basic packages"
# 注: pyext 已移除 —— 其 0.7 版使用了 Python>=3.11 删除的 inspect.getargspec,
# 无法构建, 且仅为 verl 可选工具, 训练/推理均不依赖
# numpy 钉在 2.2.6: 同时满足 numba<2.3 / mistral-common<2.4 / opencv>=2,
# 也避免步骤 1(vllm) 与步骤 5(opencv) 的依赖解析反复改写 numpy 版本
"${PIP}" install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy==2.2.6" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff

"${PIP}" install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# wheel 按实际 Python 版本选择(官方脚本固定 cp310, 仅适用于 py3.10);
# 服务器连不上 GitHub 时, 可在任何能访问的机器下载同名 wheel 放到
# 仓库根目录, 重跑本脚本会跳过下载直接安装。
PYTAG=$("${PY}" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
FAILED=""

FA_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-${PYTAG}-${PYTAG}-linux_x86_64.whl"
if [[ ! -f "${FA_WHL}" ]]; then
    wget -nv "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FA_WHL}" || {
        echo "警告: ${FA_WHL} 下载失败(GitHub 不可达), 已跳过。" >&2
        echo "      请在有网的机器下载该文件后放到仓库根目录, 再重跑本脚本。" >&2
        FAILED="${FAILED} flash-attn"
    }
fi
[[ -n "${FAILED}" ]] || "${PIP}" install --no-cache-dir "${FA_WHL}"

# flashinfer 0.2.2.post1(vllm 0.8.3+ 不兼容 >=0.2.3, 见 vllm#15777),
# wheel 为 cp38-abi3, 各 Python 版本通用
FI_WHL="flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl"
if [[ ! -f "${FI_WHL}" ]]; then
    wget -nv "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/${FI_WHL}" || {
        echo "警告: ${FI_WHL} 下载失败(GitHub 不可达), 已跳过, 处理方式同上。" >&2
        FAILED="${FAILED} flashinfer"
    }
fi
if [[ -f "${FI_WHL}" ]]; then
    "${PIP}" install --no-cache-dir "${FI_WHL}"
fi


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    NVTE_FRAMEWORK=pytorch "${PIP}" install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1
    "${PIP}" install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2
fi


echo "5. May need to fix opencv"
# headless 是 vllm->mistral_common 的依赖, 与完整版一并对齐到 4.x
# (4.14 要求 numpy>=2, 与步骤 2 的 numpy==2.2.6 兼容)
"${PIP}" install "opencv-python<5" "opencv-python-headless<5"
"${PIP}" install opencv-fixer && \
    "${PY}" -c "from opencv_fixer import AutoFix; AutoFix()"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    "${PIP}" install nvidia-cudnn-cu12==9.8.0.87
fi

if [[ -n "${FAILED}" ]]; then
    echo "以下组件未安装成功(其余已就绪): ${FAILED}" >&2
    echo "处理完后重跑本脚本即可, 已装好的部分会自动跳过。" >&2
    exit 1
fi
echo "Successfully installed all packages"
