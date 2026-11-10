#!/bin/bash
#SBATCH --job-name=criteo-dense
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/<user>/projects/smiles/logs/criteo-dense-%j.out
#SBATCH --error=/home/<user>/projects/smiles/logs/criteo-dense-%j.err
set -euo pipefail
cd "$HOME/projects/smiles/cpu_smoke"
exec "$HOME/ml-env/bin/python" -u make_dense_cache.py
