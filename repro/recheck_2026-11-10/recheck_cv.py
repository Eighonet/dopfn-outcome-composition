"""Re-check of the paper's analysis on the saved Criteo CV runs (CPU only, reads saved predictions).

1. Between- vs within-risk-stratum decomposition of Q-hat for every model and design at K=1024.
2. In-context risk ranking: a risk model fitted on the same 1,024 context rows (no treatment effect at all),
   scored on the same eval rows. Tells whether the "repaired" Do-PFN beats the context's own risk score.
3. Visitors per arm in the K-curve contexts (share vs count).
"""
import glob
import os
import re
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
H18 = REPRO / "home18"
os.environ["USERPROFILE"] = str(H18)
os.environ["HOME"] = str(H18)
sys.path.insert(0, str(REPRO / "home/projects/smiles/cpu_smoke"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklift.metrics import qini_auc_score

from fm_dense_cache import cached_dense_load

OUT = Path(sys.argv[1])
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
RESULTS = H18 / "projects/smiles/results"
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}
NB = 10

x, t, y, _, _ = cached_dense_load(DATA_ROOT)
x = np.asarray(pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)
print("loaded", x.shape, flush=True)

files = sorted(glob.glob(str(RESULTS / "fm_cv/criteo/fold*_eval8000_cuda0.predictions.npz")) +
               glob.glob(str(RESULTS / "fm_dose/criteo/fold*_eval8000_cuda0.predictions.npz")))
first = np.load(files[0])
holdout = first["final_holdout_indices"]
eval_all = np.unique(np.concatenate([np.load(f)["eval_indices"] for f in files]))
assert not np.intersect1d(holdout, eval_all).size
r = holdout[t[holdout] == 0]
m0 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(x[r], y[r])
mu0 = np.full(len(x), np.nan)
mu0[eval_all] = m0.predict_proba(x[eval_all])[:, 1]
print("mu0 fitted on", len(r), "holdout control rows", flush=True)

rng = np.random.default_rng(0)


def q(yy, s, tt):
    return float(qini_auc_score(yy, s, tt))


def decomp(s, m, yy, tt):
    r0 = rankdata(m + 1e-12 * rng.standard_normal(len(m)), method="ordinal")
    b = np.minimum((r0 - 1) * NB // len(m), NB - 1)
    df = pd.DataFrame(dict(s=s, b=b))
    between = df.groupby("b").s.transform("mean").to_numpy()
    within = df.groupby("b").s.rank(pct=True).to_numpy()
    top = b == NB - 1
    return dict(q_between=q(yy, between, tt), q_within=q(yy, within, tt),
                q_risk_then_within=q(yy, b + 0.999 * within, tt),
                top_decile_pct=float(100 * (rankdata(s)[top].mean() - 0.5) / len(s)))


def context_controls(ctx, ev):
    xc, tc, yc = x[ctx], t[ctx], y[ctx]
    med = np.nanmedian(xc, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda a: np.where(np.isfinite(a), a, med)
    mu, sd = fill(xc).mean(0), fill(xc).std(0) + 1e-9
    zc, ze = (fill(xc) - mu) / sd, (fill(x[ev]) - mu) / sd
    out = {}
    if len(np.unique(yc)) == 2:
        out["risk_logit"] = LogisticRegression(C=1.0, max_iter=2000).fit(zc, yc).predict_proba(ze)[:, 1]
        out["risk_hgb"] = HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=20, random_state=0).fit(xc, yc).predict_proba(x[ev])[:, 1]
    probs = []
    for arm in (1, 0):
        k = tc == arm
        if len(np.unique(yc[k])) < 2:
            probs.append(np.full(len(ev), yc[k].mean())); continue
        probs.append(LogisticRegression(C=1.0, max_iter=2000).fit(zc[k], yc[k]).predict_proba(ze)[:, 1])
    out["logit_T"] = probs[0] - probs[1]
    return out


rows, cache = [], {}
for f in files:
    mm = re.match(r"fold(\d+)_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx(\d+)_seed(\d+)_", Path(f).name)
    fold, model, strategy, K, seed = int(mm[1]), mm[2], STRAT[mm[3]], int(mm[4]), int(mm[5])
    z = np.load(f); ev, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"].astype(np.float64)
    yy, tt = y[ev], t[ev]
    row = dict(model=model, strategy=strategy, K=K, fold=fold, seed=seed,
               pos_t0=int(y[ctx][t[ctx] == 0].sum()), pos_t1=int(y[ctx][t[ctx] == 1].sum()),
               n_t0=int((t[ctx] == 0).sum()), n_t1=int((t[ctx] == 1).sum()),
               qhat=q(yy, s, tt), rho_mu0=spearmanr(s, mu0[ev]).statistic)
    if K == 1024:
        row.update(decomp(s, mu0[ev], yy, tt))
        key = (strategy, fold, seed)
        if key not in cache:
            c = context_controls(ctx, ev)
            d = {}
            for name, sc in c.items():
                d[f"{name}_q"] = q(yy, sc, tt)
                d[f"{name}_rho"] = spearmanr(sc, mu0[ev]).statistic
            d["mu0_q"] = q(yy, mu0[ev], tt)
            cache[key] = d
        row.update(cache[key])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("runs", len(df), "->", OUT)
