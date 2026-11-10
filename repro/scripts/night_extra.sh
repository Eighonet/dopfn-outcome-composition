#!/bin/bash
# New experiments for the paper's novelty, next to night_gpu.sh, from 01:00 on 2026-11-08.
# (b) Do-PFN outcome-share dose-response on the report's CV folds (120 GPU jobs, two shards).
# (a) FM vs neighbour mean on the 1.3M E17 rows once night_gpu.sh has adopted E17.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$R"
source env.sh
RL="$R/home/projects/smiles/logs/rerun"; NL="$RL/night_extra.log"; START=${NIGHT_START:-202611080100}
say(){ echo "$(date '+%F %T') $*" >> "$NL"; }
say "waiting for $START"
until [ "$(date +%Y%m%d%H%M)" -ge "$START" ]; do sleep 30; done

dose(){
  local H18="$R/home18" M="$R/home18/projects/smiles/manifests" DOPY="$R/.venv-dopfn/Scripts/python.exe"
  export USERPROFILE="$(winpath "$H18")" HOME="$H18" \
         FM_OUTPUT="$H18/projects/smiles/results/fm_dose"
  local pids=()
  for s in d0 d1; do "$DOPY" -u scripts/run_dose_cv.py "$M/dose_$s.txt" > "$RL/dose_$s.log" 2>&1 & pids+=($!); done
  wait "${pids[@]}"
  say "dose jobs ok=$(cat "$RL"/dose_d*.log | grep -cE '^### .*rc=0') of 120"
  "$PY" scripts/dose_summary.py > "$RL/dose_summary.txt" 2>&1; say "dose summary rc=$?"
}

e17_a(){
  until grep -qE "E17 adopted|E17 rc=[1-9]|E17: expected|E17 adopt failed|night GPU chain finished" "$RL/night.log" 2>/dev/null; do sleep 60; done
  grep -q "E17 adopted" "$RL/night.log" || { say "E17 not adopted, (a) on E17 skipped"; return; }
  cd "$R/rerun"; source "$R/rerun_env.sh"
  REPRO_DEVICE=cpu "$PY" -u ../scripts/novelty_a.py artifacts/local/m01_e17_v001_results/20261021T204248_950468Z/predictions.csv \
    E17_criteo1300k feature_knn_K512:local_feature_mean causalpfn_native_batch:local_native_batch > "$RL/novelty_a_e17.log" 2>&1
  say "(a) on E17 rc=$?"
}

say "start"
dose & D=$!
e17_a & A=$!
wait $D $A
say "night extra finished"
