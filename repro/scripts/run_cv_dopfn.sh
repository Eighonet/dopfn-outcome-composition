#!/bin/bash
# Do-PFN runs in its own environment, as on the original cluster.
cd "/c/Users/Eighonet Aerta/Desktop/ai4s/causal_fm_project/repro"; source env.sh
export PY="$R/.venv-dopfn/Scripts/python.exe"
exec "$PY" -u scripts/run_cv.py "$@"
