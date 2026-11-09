"""Compare all CV tables of report_FM.docx with the rerun: Criteo and LZD on sklearn 1.8 folds, others from the first pass."""
import glob, json, os, re, sys, zipfile
import xml.etree.ElementTree as ET
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
body = ET.fromstring(zipfile.ZipFile('../report_FM.docx').read('word/document.xml')).find(W + 'body')
txt = lambda e: ''.join(t.text or '' for t in e.iter(W + 't')).strip()
T = [[[txt(c) for c in r.findall(W + 'tc')] for r in t.findall(W + 'tr')] for t in body.findall(W + 'tbl')]
pm = lambda s: tuple(float(x) for x in s.replace('−', '-').replace(',', '.').replace(' ', '').split('±'))
DS = {'Criteo': 'criteo', 'Hillstrom': 'hillstrom', 'LZD': 'lzd', 'Orange': 'orange', 'RetailHero': 'retailhero'}
MD = {'CausalPFN': 'causalpfn', 'Do-PFN': 'dopfn', 'TabPFN-T': 'tabpfn_t'}
ST = {'R': 'random', 'T': 'balanced_treatment', 'T+Y': 'balanced_treatment_outcome'}
cv = {}
for home, wanted in (('home', {'hillstrom', 'orange', 'retailhero'}), ('home18', {'criteo', 'lzd'})):
    for p in glob.glob(f'{home}/projects/smiles/results/fm_cv/*/fold*_eval8000_cuda0.json'):
        r = json.load(open(p, encoding='utf8'))
        if r.get('metrics') and r['dataset'] in wanted:
            cv[r['dataset'], r['model'], r['context_size'], r['strategy'], r['fold'], r['model_seed']] = r['metrics']['qini_auc_normalized']
print('CV runs', len(cv), 'of 2700')
def fold_means(d, m, f):
    return [np.mean([f(k, s) for k in (512, 1024, 2048, 4096) for s in (42, 43, 44)]) for _ in [0]]
def report_grade(diff):
    return 'точно' if abs(diff) <= 1e-4 else 'близко' if abs(diff) <= 1e-3 else 'расходится'
print('\n(T+Y)-T, T-R, (T+Y)-R averaged over K, report vs rerun')
worst = {}
for r in T[8][1:]:
    d, m = DS[r[0]], MD[r[1]]
    out = []
    for col, (a, b) in zip(range(2, 5), (('T', 'R'), ('T+Y', 'T'), ('T+Y', 'R'))):
        folds = [np.mean([cv[d, m, k, ST[a], f, s] - cv[d, m, k, ST[b], f, s] for k in (512, 1024, 2048, 4096) for s in (42, 43, 44)]) for f in range(5)]
        rep = pm(r[col])[0]; got = float(np.mean(folds)); out.append(f'{a}-{b} {rep:+.5f}/{got:+.5f}')
        worst[m] = max(worst.get(m, 0), abs(got - rep))
    print(f'{r[0]:10s} {r[1]:10s} ' + ' | '.join(out))
print('worst |diff| by model:', {k: round(v, 5) for k, v in worst.items()})
print('\nK effect 4096-512')
wk = {}
for r in T[9][1:]:
    d, m = DS[r[0]], MD[r[1]]
    folds = [np.mean([cv[d, m, 4096, st, f, s] for st in ST.values() for s in (42, 43, 44)]) - np.mean([cv[d, m, 512, st, f, s] for st in ST.values() for s in (42, 43, 44)]) for f in range(5)]
    rep = pm(r[2])[0]; got = float(np.mean(folds)); wk[m] = max(wk.get(m, 0), abs(got - rep))
print('worst |diff| by model:', {k: round(v, 5) for k, v in wk.items()})
print('\nbest of 12 by CV Qini')
same = 0
for r in T[6][1:]:
    d, m = DS[r[0]], MD[r[1]]; want = r[2].replace(' ', '')
    grid = {(k, s): np.mean([cv[d, m, k, ST[s], f, sd] for f in range(5) for sd in (42, 43, 44)]) for k in (512, 1024, 2048, 4096) for s in ST}
    best = max(grid, key=grid.get); got = f'{best[0]}/{best[1]}'; same += got == want
    if got != want: print(f'  differs: {r[0]} {r[1]} report {want}, rerun {got}')
print(f'  same: {same}/15')
