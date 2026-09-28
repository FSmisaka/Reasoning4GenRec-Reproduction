PY ?= .venv/bin/python

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
