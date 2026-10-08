PY ?= .venv/bin/python
PIP := .venv/bin/pip
HF := .venv/bin/huggingface-cli

# pip 统一走清华镜像; 换源用 PIP_INDEX_URL=..., 置空恢复官方源
PIP_INDEX_URL ?= https://pypi.tuna.tsinghua.edu.cn/simple
PIP_FLAGS := $(if $(PIP_INDEX_URL),--index-url "$(PIP_INDEX_URL)",)

# 用户手动指定的 GPU(为空表示未指定)。SIDReasoner 的多卡脚本
# 据此区分: 非空则严格使用指定卡, 为空则自动挑选空闲卡。
SIDR_GPUS := $(CUDA_VISIBLE_DEVICES)
# 仅当 CUDA_VISIBLE_DEVICES 来自 make 命令行时才向 tmux 内层转发
# (外层 make 自动选中的单卡不算显式指定)
CVD_CMDLINE := $(if $(filter command line,$(origin CUDA_VISIBLE_DEVICES)),$(CUDA_VISIBLE_DEVICES),)

ifdef CUDA_VISIBLE_DEVICES
GPU := $(CUDA_VISIBLE_DEVICES)
else
GPU := $(shell nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
	| awk -F', *' '$$2 < 1000 && $$3 < 10 {print $$2, $$1}' \
	| sort -n | head -n 1 | awk '{print $$2}')
endif

ifneq ($(strip $(GPU)),)
$(info >>> 使用 GPU $(GPU))
export CUDA_DEVICE_ORDER = PCI_BUS_ID
export CUDA_VISIBLE_DEVICES = $(GPU)
else
MPS_OK := $(shell $(PY) -c "import sys, torch; sys.exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null && echo yes)
ifneq ($(strip $(MPS_OK)),)
$(info >>> 未找到空闲 GPU，使用 Apple MPS)
else
$(warning 未找到 GPU/MPS，回退到 CPU 运行（训练速度会明显变慢）；GPU 服务器上可手动指定：make <target> CUDA_VISIBLE_DEVICES=N)
endif
endif

.PHONY: setup download-model install-vllm install-rl-env smoke tmux \
	games-sasrec office-sasrec industrial-sasrec \
	games-sasrec-mini office-sasrec-mini industrial-sasrec-mini \
	games-tiger office-tiger industrial-tiger \
	games-caser office-caser industrial-caser \
	games-gru4rec office-gru4rec industrial-gru4rec \
	games-sidreasoner-sft office-sidreasoner-sft industrial-sidreasoner-sft \
	games-sidreasoner-activation office-sidreasoner-activation industrial-sidreasoner-activation \
	games-sidreasoner-rl office-sidreasoner-rl industrial-sidreasoner-rl \
	games-sidreasoner-eval office-sidreasoner-eval industrial-sidreasoner-eval \
	games-sidreasoner-think office-sidreasoner-think industrial-sidreasoner-think \
	games-sidreasoner-metrics office-sidreasoner-metrics industrial-sidreasoner-metrics \
	games-sidreasoner-merge office-sidreasoner-merge industrial-sidreasoner-merge \
	games-sidreasoner-rl-data office-sidreasoner-rl-data industrial-sidreasoner-rl-data \
	games-rqvae office-rqvae industrial-rqvae \
	games-rqvae-embed office-rqvae-embed industrial-rqvae-embed \
	games-rqvae-train office-rqvae-train industrial-rqvae-train \
	games-rqvae-apply office-rqvae-apply industrial-rqvae-apply \
	games-sid-switch office-sid-switch industrial-sid-switch

setup:
	$(PIP) install $(PIP_FLAGS) torch
	$(PIP) install $(PIP_FLAGS) transformers pandas numpy pyyaml tqdm peft datasets wandb

download-model:
	env HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-1.7B --local-dir /thuir/wangyiyao/Qwen3-1.7B

install-vllm:
	$(PIP) install $(PIP_FLAGS) vllm

install-rl-env:
	bash scripts/sidreasoner/install_rl_env.sh

# 在 tmux 会话中后台运行其他目标(断开 SSH 后服务器上继续执行):
#   make tmux TARGET=games-sidreasoner-sft [SESSION=sft]
# GPU 与直接运行相同: 未显式指定时由各脚本自动挑选空闲卡
# (外层 make 的自动选卡不转发, 避免内层误判为单卡指定)。
tmux:
	@test -n "$(TARGET)" || { echo "用法: make tmux TARGET=<目标> [SESSION=<会话名>]"; exit 1; }
	CUDA_VISIBLE_DEVICES="$(CVD_CMDLINE)" bash scripts/run_in_tmux.sh "$(TARGET)" "$(SESSION)"

games-sasrec:
	DATASET=games $(PY) -m src.train.sasrec

office-sasrec:
	DATASET=office $(PY) -m src.train.sasrec

games-sasrec-mini:
	DATASET=games $(PY) -m src.train.sasrec_mini

office-sasrec-mini:
	DATASET=office $(PY) -m src.train.sasrec_mini

games-tiger:
	DATASET=games $(PY) -m src.train.tiger

office-tiger:
	DATASET=office $(PY) -m src.train.tiger

games-caser:
	DATASET=games $(PY) -m src.train.caser

office-caser:
	DATASET=office $(PY) -m src.train.caser

industrial-sasrec:
	DATASET=industrial $(PY) -m src.train.sasrec

industrial-sasrec-mini:
	DATASET=industrial $(PY) -m src.train.sasrec_mini

industrial-tiger:
	DATASET=industrial $(PY) -m src.train.tiger

industrial-caser:
	DATASET=industrial $(PY) -m src.train.caser

games-gru4rec:
	DATASET=games $(PY) -m src.train.gru4rec

office-gru4rec:
	DATASET=office $(PY) -m src.train.gru4rec

industrial-gru4rec:
	DATASET=industrial $(PY) -m src.train.gru4rec

# SIDReasoner: 目标名 <dataset>-sidreasoner-<step>
#   sft / activation / rl / eval  -> scripts/sidreasoner/<step>.sh
#   think / rl-data               -> 直接调 python 模块(MODEL= 可覆盖待评模型)
#   metrics                       -> 对最新 think 结果计算 HR@K / NDCG@K
#   merge                         -> 合并 Stage 3 的 FSDP 分片 checkpoint
define SIDREASONER_TARGET
$(1)-sidreasoner-$(3):
	DATASET=$(1) CUDA_VISIBLE_DEVICES="$(SIDR_GPUS)" bash scripts/sidreasoner/$(2).sh
endef

define SIDREASONER_PY_TARGET
$(1)-sidreasoner-$(3):
	DATASET=$(1) MODEL=$$$${MODEL:-} $(PY) -m $(2)
endef

define SIDREASONER_METRICS_TARGET
$(1)-sidreasoner-metrics:
	@R="$$$${RESULT:-$$$$(ls -t results/sidreasoner/$(2)_think_*.json 2>/dev/null | head -n 1)}"; \
	if [ -z "$$$${R}" ]; then \
		echo "未找到 results/sidreasoner/$(2)_think_*.json, 先运行 make $(1)-sidreasoner-think"; \
		exit 1; \
	fi; \
	echo "metrics: $$$${R}"; \
	DATASET=$(1) $(PY) -m src.eval.sidreasoner_metrics calc --path "$$$${R}"
endef

define SIDREASONER_MERGE_TARGET
$(1)-sidreasoner-merge:
	CKPT_ROOT="$$$${CKPT_ROOT:-checkpoints/RecRL_Reasoning/$(2)_stage3_rl_Qwen3-1.7B}" \
	EVAL_INTERVAL="$$$${EVAL_INTERVAL:-100}" \
	bash scripts/sidreasoner/merge_ckpt_all.sh
endef

$(eval $(call SIDREASONER_TARGET,games,sft,sft))
$(eval $(call SIDREASONER_TARGET,office,sft,sft))
$(eval $(call SIDREASONER_TARGET,industrial,sft,sft))
$(eval $(call SIDREASONER_TARGET,games,activation,activation))
$(eval $(call SIDREASONER_TARGET,office,activation,activation))
$(eval $(call SIDREASONER_TARGET,industrial,activation,activation))
$(eval $(call SIDREASONER_TARGET,games,rl,rl))
$(eval $(call SIDREASONER_TARGET,office,rl,rl))
$(eval $(call SIDREASONER_TARGET,industrial,rl,rl))
$(eval $(call SIDREASONER_TARGET,games,eval,eval))
$(eval $(call SIDREASONER_TARGET,office,eval,eval))
$(eval $(call SIDREASONER_TARGET,industrial,eval,eval))

$(eval $(call SIDREASONER_PY_TARGET,games,src.eval.sidreasoner_eval_think,think))
$(eval $(call SIDREASONER_PY_TARGET,office,src.eval.sidreasoner_eval_think,think))
$(eval $(call SIDREASONER_PY_TARGET,industrial,src.eval.sidreasoner_eval_think,think))
$(eval $(call SIDREASONER_PY_TARGET,games,src.data.sidreasoner_rl_data,rl-data))
$(eval $(call SIDREASONER_PY_TARGET,office,src.data.sidreasoner_rl_data,rl-data))
$(eval $(call SIDREASONER_PY_TARGET,industrial,src.data.sidreasoner_rl_data,rl-data))

$(eval $(call SIDREASONER_METRICS_TARGET,games,Video_Games))
$(eval $(call SIDREASONER_METRICS_TARGET,office,Office_Products))
$(eval $(call SIDREASONER_METRICS_TARGET,industrial,Industrial_and_Scientific))

$(eval $(call SIDREASONER_MERGE_TARGET,games,Video_Games))
$(eval $(call SIDREASONER_MERGE_TARGET,office,Office_Products))
$(eval $(call SIDREASONER_MERGE_TARGET,industrial,Industrial_and_Scientific))

# RQ-VAE 自建 SID: 目标名 <dataset>-rqvae-<step>
#   embed: item.json 文本 -> BGE 向量(runs/rqvae/<Category>/embeddings.npy)
#   train: 训练 RQ-VAE -> 自建 SID 表(runs/rqvae/<Category>/index.own.json)
#   apply: 备份官方文件后级联替换 data/raw/Amazon 下的 4 处 SID 文件
# <dataset>-rqvae 为三步连跑。
define RQVAE_TARGET
$(1)-rqvae-$(3):
	DATASET=$(1) HF_ENDPOINT=$$$${HF_ENDPOINT:-https://hf-mirror.com} $(PY) -m $(2)
endef

$(eval $(call RQVAE_TARGET,games,src.data.rqvae_embed,embed))
$(eval $(call RQVAE_TARGET,office,src.data.rqvae_embed,embed))
$(eval $(call RQVAE_TARGET,industrial,src.data.rqvae_embed,embed))
$(eval $(call RQVAE_TARGET,games,src.train.rqvae,train))
$(eval $(call RQVAE_TARGET,office,src.train.rqvae,train))
$(eval $(call RQVAE_TARGET,industrial,src.train.rqvae,train))
$(eval $(call RQVAE_TARGET,games,src.data.rqvae_apply,apply))
$(eval $(call RQVAE_TARGET,office,src.data.rqvae_apply,apply))
$(eval $(call RQVAE_TARGET,industrial,src.data.rqvae_apply,apply))

games-rqvae: games-rqvae-embed games-rqvae-train games-rqvae-apply
office-rqvae: office-rqvae-embed office-rqvae-train office-rqvae-apply
industrial-rqvae: industrial-rqvae-embed industrial-rqvae-train industrial-rqvae-apply

# SID 集切换(官方 <-> 自建), SID=official|own 必选
games-sid-switch office-sid-switch industrial-sid-switch:
	DATASET=$(@:-sid-switch=) SID=$${SID:?需要 SID=official|own} $(PY) -m src.data.sid_switch

smoke:
	$(PY) tests/test_smoke.py
