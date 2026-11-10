#!/bin/bash
#SBATCH --job-name=smiles-preflight
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/<user>/projects/smiles/logs/preflight-%j.out
#SBATCH --error=/home/<user>/projects/smiles/logs/preflight-%j.err
set -uo pipefail
cd "$HOME/projects/smiles/cpu_smoke" || exit 2
LOGDIR="$HOME/projects/smiles/results/preflight"
mkdir -p "$LOGDIR"
failed=0
for dataset in retailhero orange lzd criteo; do
  echo "===== PREFLIGHT $dataset: context4096; balanced_treatment_outcome ====="
  log="$LOGDIR/${dataset}_ctx4096_balanced_treatment_outcome.log"
  if "$HOME/ml-env/bin/python" fm_benchmark.py \
    --dataset "$dataset" --fold 0 --seed 42 \
    --context-size 4096 --strategy balanced_treatment_outcome \
    --eval-max-size 8000 --preview > "$log" 2>&1; then
    grep -E '"(status|n_rows|n_train|n_eval|n_final_holdout|n_features|split_method|context_counts)"' "$log" || true
    echo "OK ($dataset); details: $log"
  else
    echo "CHECK REQUIRED ($dataset); last lines from $log"
    tail -n 12 "$log"
    failed=1
  fi
  echo
 done
if [ "$failed" -ne 0 ]; then
  echo "One or more preflights failed. This may indicate too few T x Y observations for 4096; inspect before launching that strategy."
  exit 1
fi
