#!/bin/bash
#SBATCH --job-name=smiles-fm-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx_2080_ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/<user>/projects/smiles/logs/fm-array-%A_%a.out
#SBATCH --error=/home/<user>/projects/smiles/logs/fm-array-%A_%a.err
set -euo pipefail
: "${MANIFEST:?Pass MANIFEST=/absolute/path via sbatch --export}"
: "${SLURM_ARRAY_TASK_ID:?Submit using sbatch --array=0-N%1}"
if [ ! -f "$MANIFEST" ]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 2
fi
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$line" ]; then
  echo "Empty manifest entry for task $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
read -r DATASET MODEL FOLD SEED CONTEXT_SIZE STRATEGY EVAL_MAX extra <<< "$line"
if [ -n "${extra:-}" ] || [ -z "${EVAL_MAX:-}" ]; then
  echo "Invalid manifest row: $line" >&2
  exit 2
fi
export DATASET MODEL FOLD SEED CONTEXT_SIZE STRATEGY EVAL_MAX
printf 'task=%s dataset=%s model=%s fold=%s seed=%s context=%s strategy=%s eval=%s\n' \
  "$SLURM_ARRAY_TASK_ID" "$DATASET" "$MODEL" "$FOLD" "$SEED" "$CONTEXT_SIZE" "$STRATEGY" "$EVAL_MAX"
exec bash "$HOME/projects/smiles/cpu_smoke/run_fm_cv.sbatch"
