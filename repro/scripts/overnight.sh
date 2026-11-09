#!/bin/bash
# Reproduce report_FM (CV, holdout, bootstrap) and M01/E19. Safe to restart: finished runs are skipped.
R0="/c/Users/Eighonet Aerta/Desktop/ai4s/causal_fm_project/repro"; cd "$R0"
touch keep_awake.flag
powershell.exe -NoProfile -Command '$k=Add-Type -MemberDefinition "[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);" -Name K -Namespace W -PassThru; [void]$k::SetThreadExecutionState(2147483649); while(Test-Path "C:\Users\Eighonet Aerta\Desktop\ai4s\causal_fm_project\repro\keep_awake.flag"){Start-Sleep 60}' &
source env.sh
export R_WIN='C:\Users\Eighonet Aerta\Desktop\ai4s\causal_fm_project\repro' CODE_WIN='C:\Users\Eighonet Aerta\Desktop\ai4s\causal_fm_project\repro\home\projects\smiles\cpu_smoke'
L=home/projects/smiles/logs; M=home/projects/smiles/manifests; DOPY="$R/.venv-dopfn/Scripts/python.exe"
DR='C:\Users\Eighonet Aerta\Desktop\ai4s\causal_fm_project\data_A_cleaned'
say(){ echo "$(date '+%F %T') $*" >> $L/overnight.log; }
wait_for(){ until grep -q "$2" "$1" 2>/dev/null; do sleep 30; done; }
wait_for $L/tabpfn_scan.log DONE
"$PY" -m pip freeze > env_main.txt; "$DOPY" -m pip freeze > env_dopfn.txt
say "start: tabpfn $(grep -i '^tabpfn==' env_main.txt)"
( cd SMILES-HypeCheck && "$PY" -u tools/run_m01_e19_v002.py all --data-root "$DR" > "$R/$L/e19.log" 2>&1; echo "E19 rc=$?" >> "$R/$L/overnight.log" ) &
E19=$!
manifest(){ ( cd "$CODE" && "$PY" build_manifest.py --datasets "$1" --models causalpfn,tabpfn_t,dopfn --folds 0,1,2,3,4 --seeds 42,43,44 \
   --contexts 512,1024,2048,4096 --strategies random,balanced_treatment,balanced_treatment_outcome --eval-max-size 8000 \
   --manifest "$R/$M/$2" --skip-completed ) >> $L/overnight.log 2>&1; }
shards(){ # file main_workers dopfn_workers
  local pids=(); rm -f $M/s_$1_*
  grep -v ' dopfn ' $M/$1 | awk -v n=$2 -v f="$M/s_$1_m" '{print > (f (NR%n) ".txt")}'
  grep ' dopfn ' $M/$1 | awk -v n=$3 -v f="$M/s_$1_d" '{print > (f (NR%n) ".txt")}'
  for f in $M/s_$1_m*.txt; do [ -f "$f" ] && { "$PY" -u scripts/run_cv.py "$f" > "$L/$(basename $f .txt).log" 2>&1 & pids+=($!); }; done
  for f in $M/s_$1_d*.txt; do [ -f "$f" ] && { "$DOPY" -u scripts/run_cv.py "$f" > "$L/$(basename $f .txt).log" 2>&1 & pids+=($!); }; done
  wait "${pids[@]}"; }
rm -f $M/cv_a.txt; manifest hillstrom,retailhero,orange,lzd cv_a.txt
say "CV without Criteo: $(wc -l < $M/cv_a.txt) jobs"; shards cv_a.txt 4 2; say "CV without Criteo finished"
wait_for $L/caches.log DONE
grep -q 'dense rc=0' $L/caches.log || { say "STOP: Criteo cache failed"; rm -f keep_awake.flag; exit 1; }
rm -f $M/cv_b.txt; manifest criteo cv_b.txt
say "CV Criteo: $(wc -l < $M/cv_b.txt) jobs"; shards cv_b.txt 4 2; say "CV Criteo finished"
rm -f $M/cv_c.txt; manifest hillstrom,retailhero,orange,lzd,criteo cv_c.txt
grep -q . $M/cv_c.txt 2>/dev/null && true; if [ -s $M/cv_c.txt ]; then say "retry: $(wc -l < $M/cv_c.txt) jobs"; shards cv_c.txt 2 1; fi
say "holdout start"
"$PY" -u scripts/run_holdout.py "$R_WIN\home\projects\smiles\manifests\holdout_frozen_20261021_233147\frozen_holdout_manifest.txt" 4 >> $L/overnight.log 2>&1
say "holdout finished: $(grep -c 'rc=0' $L/holdout.log)/48"
( cd "$CODE" && "$PY" -u bootstrap_criteo_dopfn.py --reps 1000 ) > $L/bootstrap-repro.out 2>&1
say "bootstrap rc=$?"
wait $E19
say "ALL DONE"; rm -f keep_awake.flag
