#!/bin/bash
#SBATCH --job-name=smiles-holdout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/<user>/projects/smiles/logs/holdout-%A_%a.out
#SBATCH --error=/home/<user>/projects/smiles/logs/holdout-%A_%a.err
set -euo pipefail
cd "$HOME/projects/smiles/cpu_smoke"
: "${MANIFEST:?Set MANIFEST to frozen_holdout_manifest.txt}"
: "${PLAN_SHA256:?Set PLAN_SHA256 to SHA256 of frozen manifest}"
: "${HOLDOUT_MODE:=subset}"
: "${EVAL_MAX_SIZE:=8000}"
ACTUAL_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
if [ "$ACTUAL_SHA" != "$PLAN_SHA256" ]; then echo "Frozen manifest SHA256 mismatch" >&2; exit 1; fi
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
read -r DATASET MODEL CONTEXT STRATEGY SEED <<< "$LINE"
case "$MODEL" in
    dopfn) PYTHON="$HOME/dopfn-gpu-env/bin/python" ;;
    *) PYTHON="$HOME/ml-env/bin/python" ;;
esac
ARGS=( --dataset "$DATASET" --model "$MODEL" --context-size "$CONTEXT"
       --strategy "$STRATEGY" --seed "$SEED" --device cuda:0
       --repo "$HOME/projects/Do-PFN" --plan-sha256 "$PLAN_SHA256"
       --eval-max-size "$EVAL_MAX_SIZE" )
if [ "$HOLDOUT_MODE" = full ]; then ARGS+=(--full-holdout); fi
exec "$PYTHON" -u fm_holdout.py "${ARGS[@]}"
