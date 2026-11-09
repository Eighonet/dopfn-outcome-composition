#!/bin/bash
cd "/c/Users/Eighonet Aerta/Desktop/ai4s/causal_fm_project/repro"; source env.sh
H18="$R/home18"; export USERPROFILE='C:\Users\Eighonet Aerta\Desktop\ai4s\causal_fm_project\repro\home18' HOME="$H18"
M="$H18/projects/smiles/manifests"; L="$H18/projects/smiles/logs"; DOPY="$R/.venv-dopfn/Scripts/python.exe"
pids=(); for s in d0 d1; do "$DOPY" -u scripts/run_cv.py "$M/s_lzd_$s.txt" > "$L/retry_lzd_$s.log" 2>&1 & pids+=($!); done
wait "${pids[@]}"; echo "$(date '+%F %T') lzd dopfn retry ok=$(cat "$L"/retry_lzd_d*.log | grep -cE '^### .*rc=0')" >> "$L/rerun.log"
