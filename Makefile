PY ?= .venv/bin/python

ifdef CUDA_VISIBLE_DEVICES
GPU := $(CUDA_VISIBLE_DEVICES)
else
GPU := $(shell nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
	| awk -F', *' '$$2 < 1000 && $$3 < 10 {print $$2, $$1}' \
	| sort -n | head -n 1 | awk '{print $$2}')
endif

ifeq ($(strip $(GPU)),)
$(error 未找到空闲 GPU，可手动指定：make <target> CUDA_VISIBLE_DEVICES=N)
endif

$(info >>> 使用 GPU $(GPU))

export CUDA_DEVICE_ORDER = PCI_BUS_ID
export CUDA_VISIBLE_DEVICES = $(GPU)

.PHONY: games-sasrec office-sasrec games-sasrec-mini office-sasrec-mini games-tiger office-tiger smoke

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

smoke:
	$(PY) tests/test_smoke.py
