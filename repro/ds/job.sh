#!/bin/bash
# One reproduction task on a DataSphere job VM (Linux). Rebuilds the local layout under $W:
#   $W/causal_fm_project/{data_A_cleaned, repro/{scripts, home, home18, rerun, archived_from_rerun, rerun_compare}}
# Usage: bash job.sh <kind> key=path ... ; keys: code data dopfn cache18 weights state out manifest seed
set -uo pipefail
kind=$1; shift
declare -A A; for kv in "$@"; do A[${kv%%=*}]=${kv#*=}; done
JOB=$PWD; W=$JOB/w; P=$W/causal_fm_project; R=$P/repro; mkdir -p "$R"
log(){ echo "[$(date -u '+%F %T')] $*"; }
unpack(){ for k in "$@"; do if [ -n "${A[$k]:-}" ]; then log "unpack $k"; tar -xf "${A[$k]}" -C "$P" || exit 3; fi; done; }
link(){ [ -e "$2" ] || { mkdir -p "$(dirname "$2")"; ln -s "$1" "$2"; }; }
layout(){
  link "$P" "$R/home/projects/smiles/data"
  link "$P" "$R/home18/projects/smiles/data"
  link "$R/home/projects/smiles/models" "$R/home18/projects/smiles/models"
  link "$R/home/projects/Do-PFN" "$R/home18/projects/Do-PFN"
  link "$P/data_A_cleaned" "$R/rerun/artifacts/local/data_A_cleaned"
  mkdir -p "$R/rerun_compare" "$R/rerun/logs"
}
restore_mtimes(){   # the report's caches check source parquet size and mtime_ns; tar keeps the bytes but not the nanoseconds
  python - "$P/data_A_cleaned" "$R"/home*/projects/smiles/cache/*/*.json <<'EOF'
import json, os, sys
from pathlib import Path
data, metas = Path(sys.argv[1]), sys.argv[2:]
folders = {"criteo": "Criteo-ITE-v2.1", "lzd": "LZD"}
for meta in metas:
    info = json.load(open(meta))
    stats, dataset = info.get("source_file_stats"), info.get("dataset") or Path(meta).name.split("_")[0]
    for name, sig in (stats or {}).items():
        path = data / folders[dataset] / name
        assert path.stat().st_size == sig["size_bytes"], f"size differs: {path}"
        os.utime(path, ns=(sig["mtime_ns"], sig["mtime_ns"]))
        print("mtime restored", path.relative_to(data))
EOF
}
pack_out(){ log "pack $*"; tar -czf "${A[out]}" -C "$P" "$@" 2>/dev/null || tar -czf "${A[out]}" -C "$P" --ignore-failed-read "$@"; ls -la "${A[out]}"; }
rerun_env(){
  export HOME="$R/home" HF_HOME="$R/home/.cache/huggingface" HF_HUB_OFFLINE=1
  export REPRO_DEVICE=${REPRO_DEVICE:-cuda} REPRO_BATCH=${REPRO_BATCH:-32} PYTHONUNBUFFERED=1
  git config --global --add safe.directory '*' 2>/dev/null
}
log "kind=$kind host=$(hostname) cpus=$(nproc) python=$(python -V 2>&1)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

case $kind in
probe)
  { python -V; python -c "import torch,sklearn,numpy;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-','sklearn',sklearn.__version__,'numpy',numpy.__version__)";
    git --version || echo "NO GIT"; df -h "$JOB" | tail -1; free -g | head -2; ulimit -a | grep -i "open files";
    (curl -sS -m 10 -o /dev/null -w "pypi %{http_code}\n" https://pypi.org/simple/ || echo "no internet"); } > "${A[out]}" 2>&1
  cat "${A[out]}" ;;
dose)          # Do-PFN outcome-share dose-response on the report's CV folds
  unpack code data dopfn cache18; layout; restore_mtimes || exit 4
  export HOME="$R/home18" CODE="$R/home/projects/smiles/cpu_smoke" FM_OUTPUT="$R/home18/projects/smiles/results/fm_dose"
  cd "$R"; python -u scripts/run_dose_cv.py "${A[manifest]}" ${DOSE_EXTRA:-}   # DOSE_EXTRA="--preview --device cpu" for a dry run
  pack_out repro/home18/projects/smiles/results/fm_dose ;;
holdout_share) # Do-PFN on the fixed 200k Criteo holdout subsample; runs=strategy:seed,... (protocol: repro/holdout_share_protocol.md)
  unpack code data dopfn cache18 state; layout; restore_mtimes || exit 4
  export HOME="$R/home18" CODE="$R/home/projects/smiles/cpu_smoke" HOLDOUT_OUT="$R/home18/projects/smiles/results/fm_holdout_share"
  cd "$R"
  for run in ${A[runs]//,/ }; do
    python -u scripts/run_dose_holdout.py ${run%%:*} ${run##*:} ${HS_EXTRA:-}; log "$run rc=$?"   # HS_EXTRA="--preview --device cpu" for a dry run
  done
  pack_out repro/home18/projects/smiles/results/fm_holdout_share ;;
hs10boot)      # two-level bootstrap of the 10-seed holdout check, all cores in parallel slices
  unpack code state
  export HOME="$R/home"; cd "$R"; n=$(nproc)
  for p in $(seq 0 $((n - 1))); do python -u scripts/hs10_analysis.py boot $p $n ${A[draws]:-1000} & done; wait
  python -u scripts/hs10_analysis.py report
  pack_out repro/rerun_compare/hs10_contrasts.csv ;;
leakage)       # duplicate-vector leakage on Criteo, comma-separated seeds
  unpack code data
  for s in ${A[seed]//,/ }; do
    KNN_JOBS=$(nproc) python -u "$R/scripts/novelty_c_leakage.py" "$P/data_A_cleaned" "$R/rerun_compare/novelty_c_leakage_seed$s.csv" "$s"
    log "seed $s rc=$?"
  done
  pack_out repro/rerun_compare ;;
e17chain)      # fresh M01/E17 on this GPU -> compare -> adopt (no commit) -> E18 -> analysis (a) on E17
  unpack code data weights state; layout; rerun_env; cd "$R/rerun"
  P17=projects/model_comparison/hypotheses/M01/results/E17/v001; OLD=20261021T204248_950468Z
  python -u tools/run_m01_e17_v001.py all > logs/m01_e17.log 2>&1; log "E17 rc=$?"; tail -3 logs/m01_e17.log
  NEW=$(ls $P17 | grep -E '^[0-9]{8}T' | grep -v $OLD)
  if [ $(echo "$NEW" | wc -w) -eq 1 ]; then
    python ../scripts/compare_snapshot.py "$P17/$NEW" "$R/archived_from_rerun/E17_v001/$OLD" "$R/rerun_compare/M01_E17_v001.txt" | tail -3
    ADOPT_NO_COMMIT=1 python ../scripts/adopt.py $P17 $NEW $OLD artifacts/local/m01_e17_v001_results && log "E17 adopted $NEW -> $OLD"
    python -u tools/run_m01_e18_v001.py all > logs/m01_e18.log 2>&1; log "E18 rc=$?"; tail -3 logs/m01_e18.log
    REPRO_DEVICE=cpu python -u ../scripts/novelty_a.py artifacts/local/m01_e17_v001_results/$OLD/predictions.csv E17_criteo1300k \
      feature_knn_K512:local_feature_mean causalpfn_native_batch:local_native_batch > logs/novelty_a_e17.log 2>&1; log "(a) rc=$?"
  fi
  pack_out repro/rerun/projects/model_comparison/hypotheses/M01/results/E17 repro/rerun/projects/model_comparison/hypotheses/M01/results/E18 \
           repro/rerun/artifacts/local/m01_e17_v001_results repro/rerun/artifacts/local/m01_e18_v001_results repro/rerun/logs repro/rerun_compare ;;
lzdchain)      # CS01/E04 -> adopt -> CS01/E05 -> adopt -> M01/E03 on LZD
  unpack code data weights state; layout; rerun_env; cd "$R/rerun"
  C=projects/context_selection/hypotheses/CS01/results
  step(){ local t=$1 P=$2 OLD=$3 LP=$4; shift 4
    python -u tools/run_$t.py "$@" > logs/$t.log 2>&1; local rc=$?; log "$t rc=$rc"; tail -2 logs/$t.log; [ $rc -eq 0 ] || return 1
    local NEW=$(ls $P | grep -E '^[0-9]{8}T' | grep -v $OLD); [ $(echo "$NEW" | wc -w) -eq 1 ] || return 1
    [ $t = cs01_e04 ] && python -c "import sys,pathlib;p=pathlib.Path(sys.argv[1]);p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))" "$LP/$NEW/eval_rows.csv"
    python ../scripts/compare_snapshot.py "$P/$NEW" "$R/SMILES-HypeCheck/$P/$OLD" "$R/rerun_compare/${t^^}_v001.txt" | tail -2
    ADOPT_NO_COMMIT=1 python ../scripts/adopt.py $P $NEW $OLD $LP && log "$t adopted $NEW -> $OLD"; }
  step cs01_e04 $C/E04/v001 20261018T221826_251329Z artifacts/local/cs01_e04_results &&
  step cs01_e05 $C/E05/v001 20261018T232327_431673Z artifacts/local/cs01_e05_results &&
  { python -u tools/run_m01_e03.py --source artifacts/local/cs01_e05_results/20261018T232327_431673Z > logs/m01_e03.log 2>&1; log "m01_e03 rc=$?"; }
  pack_out repro/rerun/projects/context_selection/hypotheses/CS01/results/E04 repro/rerun/projects/context_selection/hypotheses/CS01/results/E05 \
           repro/rerun/projects/model_comparison/hypotheses/M01/results/E03 repro/rerun/artifacts/local/cs01_e04_results \
           repro/rerun/artifacts/local/cs01_e05_results repro/rerun/artifacts/local/m01_e03_results repro/rerun/logs repro/rerun_compare ;;
runners)       # synthetic/PEHE rerun runners in order, on CPU with the original batch of 2; adopted after download
  unpack code data weights state; layout; rerun_env; export REPRO_DEVICE=cpu REPRO_BATCH=2; cd "$R/rerun"
  action(){ local f=tools/run_$1.py   # same rule as scripts/run_action.sh
    if grep -qE 'add_argument\("(action|seed|cohort)"' $f && grep -qE '"all"' $f; then echo all
    elif grep -qE 'add_argument\("action"' $f && grep -qE '"run"' $f; then echo run; fi; }
  outs=(repro/rerun/projects repro/rerun/logs)
  for t in ${A[runners]//,/ }; do
    python -u tools/run_$t.py $(action $t) > logs/$t.log 2>&1; log "$t rc=$?"; tail -2 logs/$t.log
    [ -d artifacts/local/${t}_results ] && outs+=(repro/rerun/artifacts/local/${t}_results)
  done
  pack_out "${outs[@]}" ;;
*) echo "unknown kind $kind"; exit 2 ;;
esac
log "done"
