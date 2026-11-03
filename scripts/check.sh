#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -v
PYTHONPATH=. python tests/smoke_train.py
PYTHONPATH=. python tests/smoke_gac_training.py
PYTHONPATH=. python tests/smoke_evaluate.py
