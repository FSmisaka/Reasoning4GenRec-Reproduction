PY ?= .venv/bin/python

PY ?= .venv/bin/python

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

.PHONY: games-sasrec office-sasrec industrial-sasrec games-sasrec-mini office-sasrec-mini industrial-sasrec-mini games-tiger office-tiger industrial-tiger games-caser office-caser industrial-caser games-gru4rec office-gru4rec industrial-gru4rec smoke

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

smoke:
	$(PY) tests/test_smoke.py
