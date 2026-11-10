#!/bin/bash
#SBATCH --job-name=smiles-fm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/<user>/projects/smiles/logs/fm-%j.out
#SBATCH --error=/home/<user>/projects/smiles/logs/fm-%j.err
set -euo pipefail
cd "$HOME/projects/smiles/cpu_smoke"
: "${DATASET:=hillstrom}"
: "${MODEL:=causalpfn}"
: "${FOLD:=0}"
: "${SEED:=42}"
: "${CONTEXT_SIZE:=512}"
: "${STRATEGY:=balanced_treatment}"
: "${EVAL_MAX:=128}"
if [ "$MODEL" = "dopfn" ]; then
  PYTHON="$HOME/dopfn-gpu-env/bin/python"
else
  PYTHON="$HOME/ml-env/bin/python"
fi
# The local checkpoint does not require a token.
unset TABPFN_TOKEN
exec "$PYTHON" -u fm_benchmark.py \
  --dataset "$DATASET" --model "$MODEL" --fold "$FOLD" --seed "$SEED" \
  --context-size "$CONTEXT_SIZE" --strategy "$STRATEGY" \
  --eval-max-size "$EVAL_MAX" --device cuda:0 \
  --data-root "$HOME/projects/smiles/data/data_A_cleaned" \
  --repo "$HOME/projects/Do-PFN" \
  --output "$HOME/projects/smiles/results/fm_cv"
