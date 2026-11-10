#!/bin/bash
# Do-PFN runs in its own environment, as on the original cluster.
cd "$(dirname "${BASH_SOURCE[0]}")/.."; source env.sh
export PY="$R/.venv-dopfn/Scripts/python.exe"
exec "$PY" -u scripts/run_cv.py "$@"
