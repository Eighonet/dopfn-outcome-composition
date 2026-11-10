"""Same re-check as recheck_cv.py + recheck_ate_sign.py, on the saved LZD CV runs (CPU only, saved predictions)."""
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

import fm_benchmark as fb

OUT = Path(sys.argv[1])
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
RESULTS = H18 / "projects/smiles/results"
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}
NB = 10

X, t, y, _, _ = fb.load(DATA_ROOT, "lzd", use_group_hashes=False)
x = np.asarray(pd.DataFrame(X).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)
print("loaded", x.shape, "treated share", t.mean().round(3), "y|t=0", y[t == 0].mean().round(4), "y|t=1", y[t == 1].mean().round(4), flush=True)

files = sorted(glob.glob(str(RESULTS / "fm_cv/lzd/fold*_eval8000_cuda0.predictions.npz")) +
               glob.glob(str(RESULTS / "fm_dose/lzd/fold*_eval8000_cuda0.predictions.npz")))
holdout = np.load(files[0])["final_holdout_indices"]
eval_all = np.unique(np.concatenate([np.load(f)["eval_indices"] for f in files]))
assert not np.intersect1d(holdout, eval_all).size
r = holdout[t[holdout] == 0]
m0 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(x[r], y[r])
mu0 = np.full(len(x), np.nan)
mu0[eval_all] = m0.predict_proba(x[eval_all])[:, 1]
print("mu0 fitted on", len(r), "holdout control rows; holdout", len(holdout), flush=True)
rng = np.random.default_rng(0)


def q(yy, s, tt):
    return float(qini_auc_score(yy, s, tt))


def decomp(s, m, yy, tt):
    r0 = rankdata(m + 1e-12 * rng.standard_normal(len(m)), method="ordinal")
    b = np.minimum((r0 - 1) * NB // len(m), NB - 1)
    df = pd.DataFrame(dict(s=s, b=b))
    between = df.groupby("b").s.transform("mean").to_numpy()
    within = df.groupby("b").s.rank(pct=True).to_numpy()
    return dict(q_between=q(yy, between, tt), q_within=q(yy, within, tt), q_risk_then_within=q(yy, b + 0.999 * within, tt))


def context_controls(ctx, ev):
    xc, yc = x[ctx], y[ctx]
    med = np.nanmedian(xc, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda a: np.where(np.isfinite(a), a, med)
    mu, sd = fill(xc).mean(0), fill(xc).std(0) + 1e-9
    zc, ze = (fill(xc) - mu) / sd, (fill(x[ev]) - mu) / sd
    out = {}
    if len(np.unique(yc)) == 2:
        out["risk_logit"] = LogisticRegression(C=1.0, max_iter=2000).fit(zc, yc).predict_proba(ze)[:, 1]
        out["risk_hgb"] = HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=20, random_state=0).fit(xc, yc).predict_proba(x[ev])[:, 1]
    return out


rows, cache = [], {}
for f in files:
    mm = re.match(r"fold(\d+)_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx(\d+)_seed(\d+)_", Path(f).name)
    fold, model, strategy, K, seed = int(mm[1]), mm[2], STRAT[mm[3]], int(mm[4]), int(mm[5])
    z = np.load(f); ev, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"].astype(np.float64)
    yy, tt = y[ev], t[ev]
    row = dict(model=model, strategy=strategy, K=K, fold=fold, seed=seed, ctx_pos=y[ctx].mean(),
               qhat=q(yy, s, tt), rho=spearmanr(s, mu0[ev]).statistic, ate_hat=s.mean(), neg_share=(s < 0).mean(),
               ctx_dm=y[ctx][t[ctx] == 1].mean() - y[ctx][t[ctx] == 0].mean(),
               eval_dm=yy[tt == 1].mean() - yy[tt == 0].mean(), mu0_q=q(yy, mu0[ev], tt))
    if K == 1024:
        row.update(decomp(s, mu0[ev], yy, tt))
        key = (strategy, fold, seed)
        if key not in cache:
            cache[key] = {f"{n}_q": q(yy, sc, tt) for n, sc in context_controls(ctx, ev).items()}
        row.update(cache[key])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
order = ["R", "T", "10%", "20%", "35%", "TY", "75%"]
k = df[df.K == 1024]
g = k.groupby(["model", "strategy"])[["ctx_pos", "qhat", "rho", "ate_hat", "neg_share", "eval_dm", "q_between", "q_within",
                                      "q_risk_then_within", "mu0_q", "risk_logit_q", "risk_hgb_q"]].mean()
print(g.reindex([(m, s) for m in ["dopfn", "causalpfn", "tabpfn_t"] for s in order if (m, s) in g.index]).round(3).to_string())
for model, d in df.groupby("model"):
    print(f"{model:10s} runs={len(d)} sign(ate_hat)==sign(rho): {(np.sign(d.ate_hat) == np.sign(d.rho)).mean():.3f} "
          f"pearson(rho, Q)={np.corrcoef(d.rho, d.qhat)[0, 1]:+.2f} spearman(ate_hat, rho)={spearmanr(d.ate_hat, d.rho).statistic:+.2f}")
d = k[k.model == "dopfn"]
print("\nDo-PFN K=1024: wins against the same-rows logistic risk ranking, by design")
print(d.assign(win=d.qhat > d.risk_logit_q, diff=d.qhat - d.risk_logit_q, qpos=d.qhat > 0)
      .groupby("strategy")[["win", "diff", "qpos"]].mean().reindex(order).round(3).to_string())
