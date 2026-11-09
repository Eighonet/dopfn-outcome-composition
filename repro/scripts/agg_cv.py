"""Aggregate CV JSONs like build_smiles_report: mean over seeds per fold, then over folds."""
import glob, json, os, sys
import numpy as np
d, m, k, st = sys.argv[1:5]
base = os.path.join(os.path.expanduser(os.environ.get('FM_OUTPUT', '~/projects/smiles/results/fm_cv')), d)
rows = []
for p in glob.glob(f'{base}/fold*_{m}_{st}_ctx{k}_seed*_eval8000_cuda0.json'):
    r = json.load(open(p, encoding='utf8'))
    if r.get('metrics'):
        rows.append((r['fold'], r['model_seed'], r['metrics'], r.get('total_time_sec'), r.get('gpu_peak_allocated_gib'), r.get('context_counts')))
print(len(rows), 'runs; mean time', round(np.mean([x[3] for x in rows]), 1), 's; peak GiB', max(x[4] or 0 for x in rows), '; counts sample', rows[0][5])
for name in ('qini_auc_normalized', 'auuc', 'uplift@5%', 'uplift@10%', 'uplift@20%', 'uplift@30%'):
    fold = [np.mean([x[2][name] for x in rows if x[0] == f]) for f in sorted({x[0] for x in rows})]
    print(f'  {name:22s} mean {np.mean(fold):+.5f}  fold SD {np.std(fold, ddof=1):.5f}')
