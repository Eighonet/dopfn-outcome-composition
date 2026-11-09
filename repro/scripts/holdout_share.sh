#!/bin/bash
# Holdout check of the outcome share (protocol: repro/holdout_share_protocol.md): two waves of three GPU runs.
R="/c/Users/Eighonet Aerta/Desktop/ai4s/causal_fm_project/repro"; cd "$R"
export USERPROFILE='C:\Users\Eighonet Aerta\Desktop\ai4s\causal_fm_project\repro\home' HOME="$R/home"
export CODE="$R/home/projects/smiles/cpu_smoke" HOLDOUT_OUT="$R/home/projects/smiles/results/fm_holdout_share"
L="$R/home/projects/smiles/logs/rerun/holdout_share"; mkdir -p "$L"
for strategy in share20 share35; do
  pids=()
  for seed in 42 43 44; do
    .venv-dopfn/Scripts/python.exe -u scripts/run_dose_holdout.py $strategy $seed > "$L/${strategy}_$seed.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; echo "$(date '+%F %T') $strategy pid $p rc=$?" >> "$L/chain.log"; done
done
echo "$(date '+%F %T') ALL DONE" >> "$L/chain.log"
