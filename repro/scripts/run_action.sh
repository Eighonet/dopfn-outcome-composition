#!/bin/bash
# Print the positional action a runner needs for a full run, or nothing.
cd "$(dirname "${BASH_SOURCE[0]}")/../rerun"
# CPU wave 1 lists cs04_e01 again although its 13:54 run (device fixed to CPU in the runner) is still
# going: keep that run's log and make the duplicate only print the status.
if [ "$1" = cs04_e01 ] && ls artifacts/local/cs04_e01_results/*/status.json >/dev/null 2>&1; then
  logs="../home/projects/smiles/logs/rerun"
  cp "$logs/cs04_e01.log" "$logs/cs04_e01.first.log"; echo --status; exit 0
fi
choices=$(grep -oE 'choices=\("all"[^)]*\)|choices=\(\{?"all"' "tools/run_$1.py" | head -1)
if grep -qE 'add_argument\("(action|seed|cohort)"' "tools/run_$1.py" && grep -qE '"all"' "tools/run_$1.py"; then echo all;
elif grep -qE 'add_argument\("action"' "tools/run_$1.py" && grep -qE '"run"' "tools/run_$1.py"; then echo run; fi
