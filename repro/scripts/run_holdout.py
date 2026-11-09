"""Run the frozen holdout plan with the unmodified fm_holdout.py, a few jobs at a time."""
import hashlib, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
R = os.environ['R_WIN']; code = os.environ['CODE_WIN']; workers = int(sys.argv[2])
plan = sys.argv[1]; sha = hashlib.sha256(open(plan, 'rb').read()).hexdigest()
jobs = [l.split() for l in open(plan, encoding='utf8') if l.strip()]
jobs.sort(key=lambda j: j[0] == 'criteo')          # the short jobs go first
main_py, do_py = rf'{R}\.venv\Scripts\python.exe', rf'{R}\.venv-dopfn\Scripts\python.exe'
log = open(os.path.join(R, 'home', 'projects', 'smiles', 'logs', 'holdout.log'), 'a', encoding='utf8')
def run(j):
    d, m, k, st, s = j; t = time.time()
    cmd = [do_py if m == 'dopfn' else main_py, '-u', 'fm_holdout.py', '--dataset', d, '--model', m, '--context-size', k,
           '--strategy', st, '--seed', s, '--device', 'cuda:0', '--plan-sha256', sha, '--full-holdout']
    p = subprocess.run(cmd, cwd=code, capture_output=True, text=True, encoding='utf8', errors='replace')
    print(f'### rc={p.returncode} {time.time()-t:.0f}s {d} {m} K{k} {st} s{s}', file=log, flush=True)
    if p.returncode: print(p.stdout[-1500:], p.stderr[-1500:], file=log, flush=True)
with ThreadPoolExecutor(workers) as ex:
    list(ex.map(run, jobs))
