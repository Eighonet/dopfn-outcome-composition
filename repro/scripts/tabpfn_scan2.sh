#!/bin/bash
# Test older tabpfn releases in a separate venv, with their dependencies present.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
L=home/projects/smiles/logs/tabpfn_scan2.log; : > "$L"
[ -d .venv-tab ] || python3 -m venv .venv-tab
V=.venv-tab/Scripts/python.exe
$V -m pip install --quiet --upgrade pip >> "$L" 2>&1
$V -m pip install --quiet "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128 >> "$L" 2>&1
$V -m pip install --quiet "tabpfn==8.0.8" numpy==1.26.4 pandas==2.3.0 pyarrow==19.0.1 scipy==1.17.1 scikit-uplift==0.5.1 tqdm==4.67.1 >> "$L" 2>&1
$V -m pip install --quiet --force-reinstall --no-deps scikit-learn==1.5.2 >> "$L" 2>&1
source env.sh; export PY="$R/.venv-tab/Scripts/python.exe"
for v in 8.0.8 8.0.4 8.0.0 7.1.1 7.0.1; do
  "$PY" -m pip install --quiet --no-deps "tabpfn==$v" >> "$L" 2>&1
  export FM_OUTPUT="$HOME/projects/smiles/results/scan2_$v"
  "$PY" -u scripts/run_cv.py home/projects/smiles/manifests/check_hillstrom_tabpfn.txt > "home/projects/smiles/logs/scan2_$v.log" 2>&1
  echo "== tabpfn $v: ok=$(grep -cE '^### .*rc=0' home/projects/smiles/logs/scan2_$v.log)/15" >> "$L"
  "$PY" scripts/agg_cv.py hillstrom tabpfn_t 2048 random 2>&1 | grep -E 'qini|auuc|uplift@|rror' >> "$L"
done
echo DONE >> "$L"
