"""Criteo final holdout: negated score, same-rows risk ranking, risk alignment, predicted average effect.

Post hoc descriptive additions to the saved holdout predictions; nothing is selected on them. CPU only.
mu0 is fitted on control rows of the development part (the holdout is group-disjoint from it).
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
STRAT = {"balanced_treatment": "T", "balanced_treatment_outcome": "TY", "share20": "20%", "share35": "35%"}
NB = 10
rng = np.random.default_rng(0)

x, t, y, _, _ = cached_dense_load(DATA_ROOT)
x = np.asarray(pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)

holdout = np.load(sorted(glob.glob(str(REPRO / "home/projects/smiles/results/fm_cv/criteo/fold0_*.npz")))[0])["final_holdout_indices"]
dev = np.setdiff1d(np.arange(len(y)), holdout)
r = dev[t[dev] == 0]
m0 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(x[r], y[r])
mu0 = np.full(len(y), np.nan)
mu0[holdout] = m0.predict_proba(x[holdout])[:, 1]
print(f"holdout {len(holdout)} rows; mu0 fitted on {len(r)} development control rows", flush=True)

files = (sorted(glob.glob(str(REPRO / "home/projects/smiles/results/fm_holdout/criteo/*.npz"))) +
         sorted(glob.glob(str(REPRO / "home/projects/smiles/results/fm_holdout_share/criteo/*.npz"))) +
         sorted(glob.glob(str(REPRO / "ds/results/hs_*/unpacked/repro/home18/projects/smiles/results/fm_holdout_share/criteo/*.npz"))))


def q(yy, s, tt):
    return float(qini_auc_score(yy, s, tt))


def decomp(s, m, yy, tt):
    r0 = rankdata(m + 1e-12 * rng.standard_normal(len(m)), method="ordinal")
    b = np.minimum((r0 - 1) * NB // len(m), NB - 1)
    df = pd.DataFrame(dict(s=s, b=b))
    between = df.groupby("b").s.transform("mean").to_numpy()
    within = df.groupby("b").s.rank(pct=True).to_numpy()
    return dict(q_between=q(yy, between, tt), q_within=q(yy, within, tt))


def risk_same_rows(ctx, ev):
    """Risk-only ranking fitted on the context rows; the treatment column is not used."""
    xc, yc = x[ctx], y[ctx]
    med = np.nanmedian(xc, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda a: np.where(np.isfinite(a), a, med)
    mu, sd = fill(xc).mean(0), fill(xc).std(0) + 1e-9
    return LogisticRegression(C=1.0, max_iter=2000).fit((fill(xc) - mu) / sd, yc).predict_proba((fill(x[ev]) - mu) / sd)[:, 1]


rows, eval_sets = [], {}
for f in files:
    mm = re.match(r"holdout_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx(\d+)_seed(\d+)_eval(all|\d+)_", Path(f).name)
    model, strategy, K, seed, size = mm[1], STRAT[mm[2]], int(mm[3]), int(mm[4]), mm[5]
    z = np.load(f)
    ev, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"].astype(np.float64)
    yy, tt = y[ev], t[ev]
    assert np.array_equal(yy, z["y"]) and np.array_equal(tt, z["t"]) and np.isin(ev, holdout).all()
    assert not np.isin(ctx, holdout).any()
    key = (len(ev), int(ev[:1000].sum()))
    eval_sets.setdefault(key, len(eval_sets))
    yc, tc = y[ctx], t[ctx]
    n1, n0, p1, p0 = (tc == 1).sum(), (tc == 0).sum(), yc[tc == 1].mean(), yc[tc == 0].mean()
    sc = risk_same_rows(ctx, ev)
    row = dict(source="ds" if "hs_" in f else "home", model=model, strategy=strategy, K=K, seed=seed, n_eval=len(ev),
               eval_set=eval_sets[key], ctx_pos=yc.mean(), ctx_dm=p1 - p0,
               ctx_se=np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0),
               q=q(yy, s, tt), q_neg=q(yy, -s, tt), rho=spearmanr(s, mu0[ev]).statistic, ate_hat=s.mean(),
               eval_dm=yy[tt == 1].mean() - yy[tt == 0].mean(), mu0_q=q(yy, mu0[ev], tt),
               risk_q=q(yy, sc, tt), risk_rho=spearmanr(sc, mu0[ev]).statistic)
    row.update(decomp(s, mu0[ev], yy, tt))
    rows.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

df = pd.DataFrame(rows)
df["z_ate"] = (df.ate_hat - df.ctx_dm) / df.ctx_se
df.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
print("\ndistinct eval sets:", len(eval_sets))
g = df.groupby(["n_eval", "source", "model", "strategy", "K"]).agg(
    n=("q", "size"), q=("q", "mean"), q_sd=("q", "std"), q_neg=("q_neg", "mean"), risk_q=("risk_q", "mean"), mu0_q=("mu0_q", "mean"),
    rho=("rho", "mean"), ate_hat=("ate_hat", "mean"), eval_dm=("eval_dm", "mean"), z_ate=("z_ate", "mean"),
    q_between=("q_between", "mean"), q_within=("q_within", "mean"),
    below_risk=("q", lambda v: int((v < df.loc[v.index, "risk_q"]).sum())),
    neg_below_risk=("q_neg", lambda v: int((v < df.loc[v.index, "risk_q"]).sum())))
print(g.round(3).to_string())
d = df[(df.source == "ds") & (df.model == "dopfn")].pivot(index="seed", columns="strategy", values=["q", "risk_q", "rho", "ate_hat"])
print("\n200k subsample, ten contexts: T+Y against 20%, and the risk ranking fitted on each context")
print(d.round(3).to_string())
