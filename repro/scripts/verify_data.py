"""Verify data_A_cleaned against recorded hashes and describe each dataset."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
root = Path('data_A_cleaned')
audit = json.loads(Path('repro/e19_source_audit.json').read_text(encoding='utf8'))
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while c := f.read(1 << 24):
            h.update(c)
    return h.hexdigest()
expected = {}
for d in audit['datasets']:
    name = d.get('dataset') or d.get('name')
    for k, v in d.get('source_sha256', {}).items():
        expected[(name, k)] = v
print('audit keys:', sorted({k[0] for k in expected}), '| files per dataset:', sorted({k[1] for k in expected}))
OUT = {'Hillstrom': 'visit', 'Retailhero-uplift': 'Y', 'LZD': 'Y', 'Criteo-ITE-v2.1': 'visit', 'Orange Telecom Churn': 'churn'}
for name in OUT:
    b = root / name
    print(f'\n=== {name}')
    print('cleaning:', json.dumps(json.loads((b / 'cleaning_metadata.json').read_text(encoding='utf8')), ensure_ascii=False)[:900])
    for fn in ('features.parquet', 'outcomes.parquet', 'cleaning_metadata.json'):
        got = sha(b / fn); exp = expected.get((name, fn))
        print(f'  {fn}: {"MATCH" if exp == got else ("MISMATCH" if exp else "no record")}  {got[:16]}')
    f, o = pd.read_parquet(b / 'features.parquet'), pd.read_parquet(b / 'outcomes.parquet')
    print('  rows', len(f), len(o), '| feature cols', f.shape[1], list(f.columns)[:16], '| outcome cols', list(o.columns)[:10])
    tcol = 'T' if 'T' in f else ('T' if 'T' in o else None)
    t = (f if 'T' in f else o)['T'].to_numpy(); y = o[OUT[name]].to_numpy()
    if 'epk_id' in f and 'epk_id' in o and not np.array_equal(f.epk_id.to_numpy(), o.epk_id.to_numpy()):
        print('  NOTE: row order differs between tables')
    r = lambda m: float(y[m].mean())
    print(f'  T share {t.mean():.4f} | Y rate {y.mean():.4f} | Y|T=1 {r(t==1):.4f} | Y|T=0 {r(t==0):.4f} | raw ATE {r(t==1)-r(t==0):+.4f}')
    print('  T×Y cells', {f'T{a}Y{b_}': int(((t==a)&(y==b_)).sum()) for a in (0,1) for b_ in (0,1)})
    num = f.select_dtypes('number')
    print('  NaN cols', int(f.isna().any().sum()), '| non-numeric cols', f.shape[1]-num.shape[1], [c for c in f.columns if c not in num.columns][:8])
