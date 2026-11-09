"""Run a manifest of CV jobs in one process with the unmodified fm_benchmark."""
import os, sys, time
manifest = os.path.abspath(sys.argv[1])
code = os.environ["CODE"]; os.chdir(code); sys.path.insert(0, code)
import fm_benchmark as fm
jobs = [l.split() for l in open(manifest) if l.strip()]
t0 = time.time()
for i, (d, m, f, s, k, st, ev) in enumerate(jobs, 1):
    sys.argv = ['fm_benchmark.py', '--dataset', d, '--model', m, '--fold', f, '--seed', s,
                '--context-size', k, '--strategy', st, '--eval-max-size', ev] + (['--output', os.environ['FM_OUTPUT']] if os.environ.get('FM_OUTPUT') else [])
    t = time.time(); rc = fm.main()
    print(f'### job {i}/{len(jobs)} rc={rc} {time.time()-t:.1f}s elapsed={time.time()-t0:.0f}s {d} {m} f{f} s{s} K{k} {st}', flush=True)
