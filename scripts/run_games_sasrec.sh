#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m src.train.train_sasrec_mini --category Video_Games "$@"
