#!/bin/bash
cd "/c/Users/Eighonet Aerta/Desktop/ai4s/causal_fm_project/repro"; source env.sh; cd "$CODE"
L="$R/home/projects/smiles/logs/caches.log"; : > "$L"
for d in lzd criteo; do "$PY" -u make_group_split_cache.py --dataset $d >> "$L" 2>&1; echo "split $d rc=$?" >> "$L"; done
"$PY" -u make_dense_cache.py >> "$L" 2>&1; echo "dense rc=$?" >> "$L"; echo DONE >> "$L"
