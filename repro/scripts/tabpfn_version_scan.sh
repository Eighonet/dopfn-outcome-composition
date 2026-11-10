#!/bin/bash
# Find the tabpfn release that reproduces the report's TabPFN-T values exactly.
cd "$(dirname "${BASH_SOURCE[0]}")/.."; source env.sh
L=home/projects/smiles/logs/tabpfn_scan.log; : > "$L"
for v in 9.1.0 9.0.0 8.5.0 8.4.0 8.3.0 8.2.0 8.1.0 8.0.8 8.0.0 7.1.1 7.0.1 9.2.0; do
  "$PY" -m pip install --quiet --no-deps "tabpfn==$v" >> "$L" 2>&1
  export FM_OUTPUT="$HOME/projects/smiles/results/scan_$v"
  "$PY" -u scripts/run_cv.py home/projects/smiles/manifests/check_hillstrom_tabpfn.txt > "home/projects/smiles/logs/scan_$v.log" 2>&1
  echo "== tabpfn $v: ok=$(grep -cE '^### .*rc=0' home/projects/smiles/logs/scan_$v.log)/15" >> "$L"
  "$PY" scripts/agg_cv.py hillstrom tabpfn_t 2048 random 2>&1 | grep -E 'qini|auuc|uplift@10|Error' >> "$L"
done
echo DONE >> "$L"
