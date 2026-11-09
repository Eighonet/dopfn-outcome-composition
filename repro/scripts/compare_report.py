"""Compare reproduced CV and holdout metrics with the tables in report_FM.docx."""
import glob, json, os, re, sys, zipfile
import xml.etree.ElementTree as ET
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
body = ET.fromstring(zipfile.ZipFile('../report_FM.docx').read('word/document.xml')).find(W + 'body')
txt = lambda e: ''.join(t.text or '' for t in e.iter(W + 't')).strip()
T = [[[txt(c) for c in r.findall(W + 'tc')] for r in t.findall(W + 'tr')] for t in body.findall(W + 'tbl')]
num = lambda s: float(re.match(r'^[+-]?\d*\.?\d+', s.replace('−', '-').replace(',', '.').replace(' ', '')).group())
pm = lambda s: tuple(float(x) for x in s.replace('−', '-').replace(',', '.').replace(' ', '').split('±'))
DS = {'Criteo': 'criteo', 'Hillstrom': 'hillstrom', 'LZD': 'lzd', 'Orange': 'orange', 'RetailHero': 'retailhero'}
MD = {'CausalPFN': 'causalpfn', 'Do-PFN': 'dopfn', 'TabPFN-T': 'tabpfn_t'}
ST = {'R': 'random', 'T': 'balanced_treatment', 'T+Y': 'balanced_treatment_outcome'}
home = os.path.expanduser('~/projects/smiles/results')
cv = {}
for p in glob.glob(f'{home}/fm_cv/*/fold*_eval8000_cuda0.json'):
    r = json.load(open(p, encoding='utf8'))
    if r.get('metrics'):
        cv[r['dataset'], r['model'], r['context_size'], r['strategy'], r['fold'], r['model_seed']] = r['metrics']
print('CV runs loaded', len(cv))
def cvstat(d, m, k, s, name):
    fold = [np.mean([cv[d, m, k, s, f, sd][name] for sd in (42, 43, 44)]) for f in range(5)]
    return np.mean(fold), np.std(fold, ddof=1)
hold, cvt, strat = T[4], T[6], T[8]
print('\n=== CV: 15 selected configs, Qini mean / SD, uplift@10, AUUC (repro minus report) ===')
rows = []
for r in cvt[1:]:
    d, m = DS[r[0]], MD[r[1]]; k, s = r[2].split('/'); k = int(k); s = ST[s.strip()]
    q0, sd0 = pm(r[3]); q, sd = cvstat(d, m, k, s, 'qini_auc_normalized')
    a0, u0 = num(r[4]), num(r[5]); a = cvstat(d, m, k, s, 'auuc')[0]; u = cvstat(d, m, k, s, 'uplift@10%')[0]
    rows.append((m, q - q0, a - a0, u - u0, sd0))
    print(f'{r[0]:10s} {r[1]:10s} {r[2]:13s} Q {q:+.5f} vs {q0:+.5f} d={q-q0:+.5f} | u10 d={u-u0:+.5f} | AUUC d={a-a0:+.5f} | fold SD {sd0:.5f}')
for m in ('causalpfn', 'dopfn', 'tabpfn_t'):
    x = [r for r in rows if r[0] == m]
    print(f'{m:10s} max|dQ|={max(abs(r[1]) for r in x):.5f} max|du10|={max(abs(r[3]) for r in x):.5f} max |dQ|/foldSD={max(abs(r[1])/r[4] for r in x):.2f}')
print('\n=== Best-of-12 selection by CV Qini: does it reproduce the frozen plan? ===')
same = 0
for r in cvt[1:]:
    d, m = DS[r[0]], MD[r[1]]; want = r[2].replace(' ', '')
    grid = {(k, s): cvstat(d, m, k, s, 'qini_auc_normalized')[0] for k in (512, 1024, 2048, 4096) for s in ST.values()}
    bk, bs = max(grid, key=grid.get); got = f"{bk}/{[a for a, b in ST.items() if b == bs][0]}"
    same += got == want
    if got != want: print(f'  differs {r[0]} {r[1]}: report {want}, repro {got}')
print(f'  same selection: {same}/15')
print('\n=== CV strategy contrasts (T+Y)-T across K ===')
for r in strat[1:]:
    d, m = DS[r[0]], MD[r[1]]
    pair = []
    for f in range(5):
        pair.append(np.mean([cv[d, m, k, ST['T+Y'], f, s]['qini_auc_normalized'] - cv[d, m, k, ST['T'], f, s]['qini_auc_normalized'] for k in (512, 1024, 2048, 4096) for s in (42, 43, 44)]))
    v0, sd0 = pm(r[3]); print(f'{r[0]:10s} {r[1]:10s} {np.mean(pair):+.5f} vs {v0:+.5f} d={np.mean(pair)-v0:+.5f}')
print('\n=== Holdout ===')
H = {}
for p in glob.glob(f'{home}/fm_holdout/*/holdout_*_evalall_cuda0.json'):
    r = json.load(open(p, encoding='utf8'))
    if r.get('status') == 'ok':
        H[r['dataset'], r['model'], r['context_size'], r['strategy'], r['model_seed']] = (r['metrics'], r.get('n_eval'), r.get('total_time_sec'))
print('holdout runs loaded', len(H))
for r in hold[1:]:
    d, m = DS[r[0]], MD[r[1]]; k, s = r[2].split('/'); k = int(k); s = ST[s.strip()]
    v = [H[d, m, k, s, sd][0] for sd in (42, 43, 44) if (d, m, k, s, sd) in H]
    if len(v) < 3:
        print(f'{r[0]:10s} {r[1]:10s} {r[2]:13s} {len(v)}/3 seeds'); continue
    q0, sd0 = pm(r[3]); u0 = pm(r[5])[0]; a0 = pm(r[4])[0]
    q = [x['qini_auc_normalized'] for x in v]; u = np.mean([x['uplift@10%'] for x in v]); a = np.mean([x['auuc'] for x in v])
    print(f'{r[0]:10s} {r[1]:10s} {r[2]:13s} Q {np.mean(q):+.5f} vs {q0:+.5f} d={np.mean(q)-q0:+.5f} (SD {np.std(q,ddof=1):.5f} vs {sd0:.5f}) | u10 {u:+.5f} vs {u0:+.5f} d={u-u0:+.5f}')
print('\n=== Main contrast Criteo Do-PFN K=1024 ===')
for sd in (42, 43, 44):
    a, b = (('criteo', 'dopfn', 1024, 'balanced_treatment', sd), ('criteo', 'dopfn', 1024, 'balanced_treatment_outcome', sd))
    print(sd, 'T', round(H[a][0]['qini_auc_normalized'], 5) if a in H else None, 'T+Y', round(H[b][0]['qini_auc_normalized'], 5) if b in H else None,
          'dQ', round(H[b][0]['qini_auc_normalized'] - H[a][0]['qini_auc_normalized'], 5) if a in H and b in H else None)
print('report seeds dQ: 42 +0.06933, 43 +0.09841, 44 +0.11314; mean +0.09363')
