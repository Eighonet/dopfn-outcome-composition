#!/bin/bash
# GPU tail for the night of 2026-11-08, started at 01:00 at the user's request (the GPU was handed back at 15:38).
# Line A: resume E17 -> compare with the archive -> adopt (no commit) -> E18.
# Line B: resume E19 in the original clone; after it, LZD: CS01/E04 -> adopt -> CS01/E05 -> adopt -> M01/E03.
# Nothing is committed: CPU runners are still active and packagers compare the commit.
R="/c/Users/Eighonet Aerta/Desktop/ai4s/causal_fm_project/repro"; cd "$R"
source rerun_env.sh                      # PY, RL, REPRO_DEVICE=cuda, REPRO_BATCH=32 (the batch E17 started with)
DR='C:\Users\Eighonet Aerta\Desktop\ai4s\causal_fm_project\data_A_cleaned'
NL="$RL/night.log"; START=${NIGHT_START:-202611080100}
say(){ echo "$(date '+%F %T') $*" >> "$NL"; }
newest(){ ls "$1" | grep -E '^[0-9]{8}T[0-9]{6}_[0-9]+Z$' | grep -v "$2"; }   # the one snapshot id that is not archived
adopt(){ # <snapshot parent> <new id> <archived id> <local parent>; git may briefly hold index.lock for a runner
  for try in 1 2 3; do ADOPT_NO_COMMIT=1 "$PY" ../scripts/adopt.py "$@" >> "$NL" 2>&1 && return 0; sleep 30; done; return 1; }
lf(){ "$PY" -c "import sys,pathlib;p=pathlib.Path(sys.argv[1]);p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))" "$1"; }

say "waiting for $START"
until [ "$(date +%Y%m%d%H%M)" -ge "$START" ]; do sleep 30; done
say "start; GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"

line_a(){
  cd "$R/rerun"
  "$PY" -u tools/run_m01_e17_v001.py all --resume >> "$RL/m01_e17.log" 2>&1; rc=$?; say "E17 rc=$rc"
  [ $rc -eq 0 ] || return
  P=projects/model_comparison/hypotheses/M01/results/E17/v001; OLD=20261021T204248_950468Z; NEW=$(newest $P $OLD)
  [ $(echo "$NEW" | wc -w) -eq 1 ] || { say "E17: expected one new snapshot, got '$NEW'"; return; }
  "$PY" ../scripts/compare_snapshot.py "$P/$NEW" "$R/archived_from_rerun/E17_v001/$OLD" "$R/rerun_compare/M01_E17_v001.txt" > /dev/null 2>&1
  say "E17 compare: $(tail -1 "$R/rerun_compare/M01_E17_v001.txt"); summary $(tr -d '\n ' < $P/$NEW/summary.json | cut -c1-160)"
  adopt $P $NEW $OLD artifacts/local/m01_e17_v001_results || { say "E17 adopt failed"; return; }
  say "E17 adopted $NEW -> $OLD"
  "$PY" -u tools/run_m01_e18_v001.py all > "$RL/m01_e18.log" 2>&1; say "E18 rc=$?"
}

lzd_step(){ # <runner> <snapshot parent> <archived id> <local parent> [args]
  local t=$1 P=$2 OLD=$3 LP=$4; shift 4
  "$PY" -u tools/run_$t.py "$@" > "$RL/$t.log" 2>&1; rc=$?; say "$t rc=$rc"
  [ $rc -eq 0 ] || return 1
  local NEW=$(newest $P $OLD)
  [ $(echo "$NEW" | wc -w) -eq 1 ] || { say "$t: expected one new snapshot, got '$NEW'"; return 1; }
  [ "$t" = cs01_e04 ] && lf "$LP/$NEW/eval_rows.csv"   # CS01/E05 checks these bytes against the archived LF hash
  "$PY" ../scripts/compare_snapshot.py "$P/$NEW" "$R/SMILES-HypeCheck/$P/$OLD" "$R/rerun_compare/${t^^}_v001.txt" > /dev/null 2>&1
  say "$t compare: $(tail -1 "$R/rerun_compare/${t^^}_v001.txt")"
  adopt $P $NEW $OLD $LP || { say "$t adopt failed"; return 1; }
  say "$t adopted $NEW -> $OLD"
}

line_b(){
  cd "$R/SMILES-HypeCheck"
  "$PY" -u tools/run_m01_e19_v002.py all --resume --data-root "$DR" >> "$R/home/projects/smiles/logs/e19.log" 2>&1
  say "E19 rc=$?"
  cd "$R/rerun"
  C=projects/context_selection/hypotheses/CS01/results
  lzd_step cs01_e04 $C/E04/v001 20261018T221826_251329Z artifacts/local/cs01_e04_results || return
  lzd_step cs01_e05 $C/E05/v001 20261018T232327_431673Z artifacts/local/cs01_e05_results || return
  "$PY" -u tools/run_m01_e03.py --source artifacts/local/cs01_e05_results/20261018T232327_431673Z > "$RL/m01_e03.log" 2>&1
  say "m01_e03 rc=$?"
}

line_a & A=$!
line_b & B=$!
wait $A $B
say "night GPU chain finished; GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
