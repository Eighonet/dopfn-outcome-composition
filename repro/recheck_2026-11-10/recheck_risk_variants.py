"""Does the same-rows risk reference depend on the choice of the risk model? Criteo, CPU only, no foundation model.

Usage: recheck_risk_variants.py OUT.csv
Reads the saved Do-PFN runs of Criteo: the paper's folds with the share designs (`home18`, 240 contexts) and the second
fold split (`home`, 180 contexts). Every variant is fitted on the saved context rows and scored on the fold's 8,000
evaluation rows. Risk-only rankings (the treatment column is not used):
  lr               logistic regression, median fill, standardised, C=1: the reference line of recheck_five.py
  lr_C0.01..100    the same with another C
  hgb              HistGB (lr 0.05, 200 iterations) on the raw features
  lr_ctrl, lr_trt  the reference fitted on the control / treated rows of the context only
Uplift scores from the same rows, for contrast:
  lr_tdiff         lr_trt - lr_ctrl, a logistic T-learner
  lr_s             logistic S-learner: features plus the treatment column, p(x, 1) - p(x, 0); `lr_s_b` is the
                   coefficient of the treatment column
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import glob
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
from scipy.stats import t as tdist
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklift.metrics import qini_auc_score

from fm_dense_cache import cached_dense_load

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
TREES = {"home18": ["fm_cv", "fm_dose"], "home": ["fm_cv"]}
SAVED = {"home18": "recheck_five_h18.csv", "home": "recheck_five_grouped.csv"}
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}
ORDER = ["R", "T", "10%", "20%", "35%", "TY", "75%"]
CS = [0.01, 0.1, 10.0, 100.0]
RISK = ["lr"] + [f"lr_C{c:g}" for c in CS] + ["hgb", "lr_ctrl", "lr_trt"]
UPLIFT = ["lr_tdiff", "lr_s"]


def q(yy, s, tt):
    return float("nan") if s is None else float(qini_auc_score(yy, s, tt))


def prep(xc):
    """Median fill and standardisation fitted on the given rows, as in recheck_five.py."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(xc, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda a: np.where(np.isfinite(a), a, med)
    mu, sd = fill(xc).mean(0), fill(xc).std(0) + 1e-9
    return lambda a: (fill(a) - mu) / sd


def lr(xc, yc, xe, C=1.0):
    if len(np.unique(yc)) < 2:
        return None
    f = prep(xc)
    return LogisticRegression(C=C, max_iter=2000).fit(f(xc), yc).predict_proba(f(xe))[:, 1]


def hgb(xc, yc, xe):
    if len(np.unique(yc)) < 2:
        return None
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(xc, yc).predict_proba(xe)[:, 1]


def lr_s(xc, tc, yc, xe):
    if len(np.unique(yc)) < 2 or len(np.unique(tc)) < 2:
        return None, np.nan
    f = prep(xc)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(np.column_stack([f(xc), tc]), yc)
    p = lambda arm: m.predict_proba(np.column_stack([f(xe), np.full(len(xe), arm)]))[:, 1]
    return p(1) - p(0), float(m.coef_[0, -1])


def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return (v.mean(), tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (np.nan, np.nan)


files = {tree: [f for sub in subs for f in sorted(glob.glob(str(
    REPRO / tree / "projects/smiles/results" / sub / "criteo" / "fold*_dopfn_*_eval8000_cuda0.predictions.npz")))]
    for tree, subs in TREES.items()}
x, t, y, _, _ = cached_dense_load(DATA_ROOT)
t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)
need = np.unique(np.concatenate([np.load(f)[k] for fs in files.values() for f in fs for k in ("context_indices", "eval_indices")]))
xs = np.asarray(pd.DataFrame(np.asarray(x[need])).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
rows_of = lambda idx: xs[np.searchsorted(need, idx)]
print(f"criteo: rows {len(y)}, used {len(need)}, features {xs.shape[1]}; runs {({k: len(v) for k, v in files.items()})}", flush=True)

rows = []
for tree, fs in files.items():
    for f in fs:
        mm = re.match(r"fold(\d+)_dopfn_(.+)_ctx(\d+)_seed(\d+)_", Path(f).name)
        z = np.load(f)
        ev, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"].astype(np.float64)
        yy, tt, xe = y[ev], t[ev], rows_of(ev)
        assert np.array_equal(yy, z["y"]) and np.array_equal(tt, z["t"])
        xc, yc, tc = rows_of(ctx), y[ctx], t[ctx]
        c0, c1 = tc == 0, tc == 1
        row = dict(tree=tree, strategy=STRAT[mm[2]], K=int(mm[3]), fold=int(mm[1]), seed=int(mm[4]),
                   n_ctrl=int(c0.sum()), pos_ctrl=int(yc[c0].sum()), pos_trt=int(yc[c1].sum()),
                   q=q(yy, s, tt), q_neg=q(yy, -s, tt), lr=q(yy, lr(xc, yc, xe), tt))
        for c in CS:
            row[f"lr_C{c:g}"] = q(yy, lr(xc, yc, xe, C=c), tt)
        row["hgb"] = q(yy, hgb(xc, yc, xe), tt)
        p0, p1 = lr(xc[c0], yc[c0], xe), lr(xc[c1], yc[c1], xe)
        row["lr_ctrl"], row["lr_trt"] = q(yy, p0, tt), q(yy, p1, tt)
        row["lr_tdiff"] = q(yy, None if p0 is None or p1 is None else p1 - p0, tt)
        tau, row["lr_s_b"] = lr_s(xc, tc, yc, xe)
        row["lr_s"] = q(yy, tau, tt)
        rows.append(row)
    print(f"{tree}: {len(fs)} contexts done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 250)

for tree, d in df.groupby("tree", sort=False):
    old = pd.read_csv(HERE / SAVED[tree])
    old = old[(old.ds == "criteo") & (old.model == "dopfn")][["strategy", "K", "fold", "seed", "q", "risk_q", "mu0_q"]]
    m = d.merge(old, on=["strategy", "K", "fold", "seed"], suffixes=("", "_saved"), validate="one_to_one")
    assert len(m) == len(d) == len(old)
    print(f"\n##### {tree}: {len(d)} contexts; max |Q - saved Q| = {(m.q - m.q_saved).abs().max():.1e}; "
          f"max |lr - saved same-rows risk| = {(m.lr - m.risk_q).abs().max():.1e}")
    designs = [s for s in ORDER if s in set(d.strategy)]
    cols = ["q", "q_neg", "mu0_q"] + RISK + UPLIFT
    fm = m.groupby(["strategy", "fold"])[cols].mean()            # fold means over K and seeds; missing runs are skipped
    print("Q-hat, mean of the five fold means (share designs: K=1,024 only; R, T, T+Y: all K)")
    print(fm.groupby("strategy").mean().loc[designs].round(3).to_string())
    print("runs without a score (a single class among the rows used):",
          {c: int(d[c].isna().sum()) for c in RISK + UPLIFT if d[c].isna().any()} or "none")
    print("SD of the fold means, by variant (R, T, T+Y pooled):",
          m[m.strategy.isin(["R", "T", "TY"])].groupby("fold")[RISK].mean().std().round(3).to_dict())

    print("gap to each risk-only variant: mean of the fold means of the per-run gaps [95% t-interval], folds below")
    targets = [("Do-PFN T+Y", "TY", "q"), ("Do-PFN 20%", "20%", "q"), ("Do-PFN R negated", "R", "q_neg"),
               ("Do-PFN T negated", "T", "q_neg")]
    out = {}
    for label, strategy, col in targets:
        e = m[m.strategy == strategy]
        if not len(e):
            continue
        for ref in RISK + ["mu0_q"]:
            g = (e[col] - e[ref]).groupby(e.fold).mean().dropna()
            mean, half = ci(g)
            out.setdefault(label, {})[ref] = f"{mean:+.3f} [{mean - half:+.3f}, {mean + half:+.3f}] {int((g < 0).sum())}/{len(g)}"
    print(pd.DataFrame(out).to_string())

    print("Do-PFN T+Y by K: folds (of 5) in which the fold mean is below the variant, and the mean gap")
    e = m[m.strategy == "TY"]
    byk = {}
    for K, ek in e.groupby("K"):
        for ref in RISK:
            g = (ek.q - ek[ref]).groupby(ek.fold).mean().dropna()
            byk.setdefault(K, {})[ref] = f"{int((g < 0).sum())}/{len(g)} {g.mean():+.3f}"
    print(pd.DataFrame(byk).to_string())

    print("uplift scores from the same rows: mean Q-hat [95% t-interval over folds]; logistic S-learner by coefficient sign")
    for strategy in designs:
        e = m[m.strategy == strategy]
        parts = []
        for c in UPLIFT:
            mean, half = ci(e.groupby("fold")[c].mean())
            parts.append(f"{c} {mean:+.3f} [{mean - half:+.3f}, {mean + half:+.3f}]")
        pos = e[e.lr_s_b > 0]
        parts.append(f"lr_s_b > 0 in {len(pos)}/{int(e.lr_s_b.notna().sum())} contexts; lr_s Q-hat {pos.lr_s.mean():+.3f} there, "
                     f"{e[e.lr_s_b < 0].lr_s.mean():+.3f} where it is negative")
        print(f"  {strategy:4s} " + "; ".join(parts))
