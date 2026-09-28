#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DATASET=games
exec .venv/bin/python -m src.train.caser
