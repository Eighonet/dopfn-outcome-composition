#!/bin/bash
# Redo LZD and Criteo CV on scikit-learn 1.8 folds, the ones the report used. Finished runs are skipped.
R0="/c/Users/Eighonet Aerta/Desktop/ai4s/causal_fm_project/repro"; cd "$R0"; source env.sh
H18="$R/home18"; export USERPROFILE='C:\Users\Eighonet Aerta\Desktop\ai4s\causal_fm_project\repro\home18' HOME="$H18"
L="$H18/projects/smiles/logs"; M="$H18/projects/smiles/manifests"; mkdir -p "$M"
V="$R/.venv-tab/Scripts/python.exe"; DOPY="$R/.venv-dopfn/Scripts/python.exe"
say(){ echo "$(date '+%F %T') $*" >> "$L/rerun.log"; }
say "start; split sklearn: $("$V" -c 'import sklearn;print(sklearn.__version__)')"
( cd "$CODE"
  [ -f "$H18/projects/smiles/cache/group_splits/criteo_fold_membership.json" ] || "$V" -u make_group_split_cache.py --dataset criteo >> "$L/cache.log" 2>&1
  [ -f "$H18/projects/smiles/cache/dense_features/criteo_dense_meta.json" ] || "$V" -u make_dense_cache.py >> "$L/cache.log" 2>&1 )
say "caches ready: $(ls "$H18/projects/smiles/cache/dense_features" | wc -l) dense files"
for d in lzd criteo; do
  rm -f "$M/rerun_$d.txt" "$M"/s_"$d"_*
  ( cd "$CODE" && "$PY" build_manifest.py --datasets $d --models causalpfn,tabpfn_t,dopfn --folds 0,1,2,3,4 --seeds 42,43,44 \
     --contexts 512,1024,2048,4096 --strategies random,balanced_treatment,balanced_treatment_outcome --eval-max-size 8000 \
     --manifest "$M/rerun_$d.txt" --skip-completed ) >> "$L/rerun.log" 2>&1
  grep -v ' dopfn ' "$M/rerun_$d.txt" | awk -v f="$M/s_${d}_m" '{print > (f (NR%3) ".txt")}'
  grep ' dopfn ' "$M/rerun_$d.txt" | awk -v f="$M/s_${d}_d" '{print > (f (NR%2) ".txt")}'
  pids=()
  for f in "$M"/s_"$d"_m*.txt; do "$PY" -u scripts/run_cv.py "$f" > "$L/$(basename "$f" .txt).log" 2>&1 & pids+=($!); done
  for f in "$M"/s_"$d"_d*.txt; do "$DOPY" -u scripts/run_cv.py "$f" > "$L/$(basename "$f" .txt).log" 2>&1 & pids+=($!); done
  wait "${pids[@]}"; say "$d finished: $(cat "$L"/s_${d}_*.log | grep -cE '^### .*rc=0') ok"
done
say "ALL DONE"
