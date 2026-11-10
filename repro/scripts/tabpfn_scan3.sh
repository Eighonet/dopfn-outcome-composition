#!/bin/bash
# Last check on the TabPFN gap: newer scikit-learn and numpy with tabpfn 9.2.0.
cd "$(dirname "${BASH_SOURCE[0]}")/.."; source env.sh
export PY="$R/.venv-tab/Scripts/python.exe"
L=home/projects/smiles/logs/tabpfn_scan3.log; : > "$L"
"$PY" -m pip install --quiet "tabpfn==9.2.0" >> "$L" 2>&1
for c in "1.6.1 1.26.4" "1.7.2 1.26.4" "1.7.2 2.2.6" "1.8.0 2.2.6"; do set -- $c
  "$PY" -m pip install --quiet --no-deps "scikit-learn==$1" "numpy==$2" >> "$L" 2>&1
  export FM_OUTPUT="$HOME/projects/smiles/results/scan3_$1_$2"
  "$PY" -u scripts/run_cv.py home/projects/smiles/manifests/check_hillstrom_tabpfn.txt > "home/projects/smiles/logs/scan3_$1_$2.log" 2>&1
  echo "== sklearn $1 numpy $2: ok=$(grep -cE '^### .*rc=0' home/projects/smiles/logs/scan3_$1_$2.log)/15" >> "$L"
  "$PY" scripts/agg_cv.py hillstrom tabpfn_t 2048 random 2>&1 | grep -E 'qini|uplift@10|rror' >> "$L"
done
echo DONE >> "$L"
