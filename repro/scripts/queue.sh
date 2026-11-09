#!/bin/bash
# Run a list of rerun runners, a few at a time. Usage: queue.sh <parallel> <runner> [<runner> ...]
cd "/c/Users/Eighonet Aerta/Desktop/ai4s/causal_fm_project/repro"; source rerun_env.sh; cd rerun
P=$1; shift
run_one(){ t=$1; arg=$(./../scripts/run_action.sh "$t"); "$PY" -u tools/run_$t.py $arg > "$RL/$t.log" 2>&1; echo "$t exit $? $(date '+%H:%M')" >> "$RL/chain.log"; }
export -f run_one
printf '%s\n' "$@" | xargs -P "$P" -I{} bash -c 'source "/c/Users/Eighonet Aerta/Desktop/ai4s/causal_fm_project/repro/rerun_env.sh"; cd "$R/rerun"; run_one {}'
