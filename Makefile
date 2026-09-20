PY ?= .venv/bin/python
SASREC = $(PY) -m src.train.train_sasrec_mini
TIGER = $(PY) -m src.train.train_tiger
TIGER_B1 = $(TIGER) --layers 4 --d-model 192 --heads 4 --d-ff 768 --batch-size 256 --lr 1e-3 --warmup-steps 1000 --scheduler invsq --epochs 200 --patience 8

.PHONY: games-sasrec office-sasrec games-tiger office-tiger smoke

games-sasrec:
	$(SASREC) --category Video_Games

office-sasrec:
	$(SASREC) --category Office_Products

games-tiger:
	$(TIGER_B1) --category Video_Games

office-tiger:
	$(TIGER_B1) --category Office_Products

smoke:
	$(PY) tests/test_smoke.py
