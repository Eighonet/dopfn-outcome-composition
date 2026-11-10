#!/bin/bash
# Rerun CS01/E01/v003, the source of the photo table, on the GPU with the exact archived script.
cd "$(dirname "${BASH_SOURCE[0]}")/.."; source env.sh
mkdir -p runs/cs01_e01_v003
exec "$PY" -u scripts/cs01_v003_gpu.py --run --device cuda --batch 32 --threads 2 --folds 5 \
  --datasets Hillstrom Retailhero-uplift LZD Criteo-ITE-v2.1 --context-sizes 512 --context-seeds 42 43 44 \
  --train-max 51200 --eval-max 4096 --criteo-sample-max 100000 --criteo-sample-seed 42 --split-seed 42 \
  --bootstrap 500 --effect-score-mode crossfit --nuisance-folds 5 --nuisance-seed 42 \
  --revision 83aad07da1cb077cfda4236878a1b07dc9f72a54 \
  --data-root "$(winpath "${DATA_ROOT:-$R/../data_A_cleaned}")" \
  --out-dir "$(winpath "$R/runs/cs01_e01_v003")"
