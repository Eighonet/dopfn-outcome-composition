"""Compare finished CS01 folds with the archived v003 metrics, method by method."""
import csv, glob, sys
sys.stdout.reconfigure(encoding='utf-8')
ref = {}
for r in csv.DictReader(open('SMILES-HypeCheck/projects/context_selection/hypotheses/CS01/results/E01/v003/20261013T202852_574199Z/all_metrics.csv', encoding='utf8')):
    ref[r['dataset'], int(r['fold']), r['method']] = (float(r['normalized_qini_auc']), float(r['uplift_10_pp']))
rows = []
for p in sorted(glob.glob('runs/cs01_e01_v003/*/*/fold_*/metrics.csv')):
    for r in csv.DictReader(open(p, encoding='utf8')):
        k = (r['dataset'], int(r['fold']), r['method']); q, u = float(r['normalized_qini_auc']), float(r['uplift_10_pp'])
        if k in ref: rows.append((*k, q, ref[k][0], q - ref[k][0], u - ref[k][1]))
print(len(rows), 'of', len(ref), 'metrics compared')
for r in rows[:9]: print(f'{r[0]:18s} f{r[1]} {r[2]:20s} Q {r[3]:+.5f} vs {r[4]:+.5f} d={r[5]:+.6f} | u10 d={r[6]:+.4f} pp')
if rows:
    print('max |dQ|', round(max(abs(r[5]) for r in rows), 6), ' max |du10| pp', round(max(abs(r[6]) for r in rows), 4))
