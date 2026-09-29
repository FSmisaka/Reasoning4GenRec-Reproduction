#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export DATASET=industrial
exec .venv/bin/python -m src.train.tiger
