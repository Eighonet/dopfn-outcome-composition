#!/bin/bash
# When the holdout is complete: parallel bootstrap first, then the CPU-only rerun runners at low priority.
R0="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$R0"; source env.sh
L=home/projects/smiles/logs; RLOG=$L/rerun/chain.log
lower(){ powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { (\$_.CommandLine -like '*tools/run_*' -or \$_.CommandLine -like '*bootstrap_parallel*') -and \$_.CommandLine -notlike '*run_holdout*' } | ForEach-Object { \$p=Get-Process -Id \$_.ProcessId -ErrorAction SilentlyContinue; if (\$p -and \$p.PriorityClass -ne 'BelowNormal') { try { \$p.PriorityClass='BelowNormal' } catch {} } }" >/dev/null 2>&1; }
until [ "$(grep -c 'rc=0' $L/holdout.log)" -ge 48 ]; do sleep 60; lower; done
echo "$(date '+%H:%M') holdout complete, parallel bootstrap starts" >> $RLOG
"$PY" -u scripts/bootstrap_parallel.py --reps 1000 --workers 12 > $L/bootstrap-parallel.out 2>&1
echo "bootstrap_parallel exit $? $(date '+%H:%M')" >> $RLOG
export REPRO_DEVICE=cpu REPRO_BATCH=2
( while pgrep -f xargs >/dev/null 2>&1 || true; do lower; sleep 60; [ -f "$R0/.cpu_wave_done" ] && break; done ) &
bash scripts/queue.sh 6 m01_e12 m01_e13 m06_e02 m06_e03 m06_e08 m06_e10 m06_e11 m05_e01 cs07_e01 cs04_e01
touch "$R0/.cpu_wave_done"; echo "cpu wave 1 finished $(date '+%H:%M')" >> $RLOG
