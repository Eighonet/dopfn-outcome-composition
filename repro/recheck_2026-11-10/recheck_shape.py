"""Shape of the effect score along baseline risk: is tau-hat a constant shift on the log-odds scale?

If the model predicts logit(p1) = logit(p0) + c, then tau-hat is about c * p(1 - p), with p the risk on the scale of the
context (after the outcome enrichment). That gives a sign rule with a mirror at p = 0.5 inside a single run: the score
follows risk below 0.5 and runs against it above. Saved predictions only, CPU only.

p on the context scale is taken two ways: mu0-hat shifted to the control-arm share of positives in the context
(`shift`), and a logistic model fitted on the context rows without the treatment column (`logit`).
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
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import fm_benchmark as fb
from fm_dense_cache import cached_dense_load

OUT = Path(sys.argv[1])
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
# the paper's folds for Criteo and LZD (with the share designs), the only saved folds for the other three
SOURCES = {"criteo": [("home18", "fm_cv"), ("home18", "fm_dose")], "lzd": [("home18", "fm_cv"), ("home18", "fm_dose")],
           "hillstrom": [("home", "fm_cv")], "retailhero": [("home", "fm_cv")], "orange": [("home", "fm_cv")]}
SETS = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SOURCES)
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}
ORDER = ["R", "T", "10%", "20%", "35%", "TY", "75%"]
NB = 10


def numeric(X):
    """Numeric feature matrix: numbers as they are, short categoricals one-hot, long ones as codes."""
    X = pd.DataFrame(X)
    parts = []
    for c in X.columns:
        col = X[c]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            parts.append(pd.to_numeric(col, errors="coerce").astype(np.float64).rename(str(c)))
        else:
            cat = col.astype("category")
            if cat.cat.categories.size <= 30:
                parts.append(pd.get_dummies(cat, prefix=str(c), dtype=np.float64))
            else:
                parts.append(pd.Series(cat.cat.codes.astype(np.float64), name=str(c), index=X.index).replace(-1.0, np.nan))
    return np.asarray(pd.concat(parts, axis=1), dtype=np.float64)


def risk_same_rows(x, y, ctx, ev):
    """Risk-only model fitted on the context rows; the treatment column is not used."""
    xc, yc = x[ctx], y[ctx]
    if len(np.unique(yc)) < 2:
        return None
    with np.errstate(all="ignore"):
        med = np.nanmedian(xc, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda a: np.where(np.isfinite(a), a, med)
    mu, sd = fill(xc).mean(0), fill(xc).std(0) + 1e-9
    return LogisticRegression(C=1.0, max_iter=2000).fit((fill(xc) - mu) / sd, yc).predict_proba((fill(x[ev]) - mu) / sd)[:, 1]


def sp(a, b):
    return np.nan if len(a) < 200 or np.ptp(a) == 0 or np.ptp(b) == 0 else spearmanr(a, b).statistic


rows = []
for ds in SETS:
    if ds == "criteo":
        x, t, y, _, _ = cached_dense_load(DATA_ROOT)
        x = np.asarray(pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
    else:
        X, t, y, _, _ = fb.load(DATA_ROOT, ds, use_group_hashes=False)
        x = numeric(X)
    t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)
    files = [f for tree, sub in SOURCES[ds]
             for f in sorted(glob.glob(str(REPRO / tree / "projects/smiles/results" / sub / ds / "fold*_eval8000_cuda0.predictions.npz")))]
    holdout = np.load(files[0])["final_holdout_indices"]
    eval_sets = {}
    for f in files:
        fold = int(re.match(r"fold(\d+)_", Path(f).name)[1])
        if fold not in eval_sets:
            eval_sets[fold] = np.load(f)["eval_indices"]
    eval_all = np.unique(np.concatenate(list(eval_sets.values())))
    assert not np.intersect1d(holdout, eval_all).size
    pool = holdout if ds in ("criteo", "lzd") else np.setdiff1d(np.arange(len(y)), eval_all)
    r = pool[t[pool] == 0]
    m0 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(x[r], y[r])
    mu0 = np.full(len(y), np.nan)
    mu0[eval_all] = np.clip(m0.predict_proba(x[eval_all])[:, 1], 1e-6, 1 - 1e-6)
    base0 = y[t == 0].mean()
    print(f"=== {ds}: {len(files)} runs, control base rate {base0:.4f}, mu0 on {len(r)} control rows", flush=True)

    cache = {}
    for f in files:
        mm = re.match(r"fold(\d+)_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx(\d+)_seed(\d+)_", Path(f).name)
        fold, model, strategy, K, seed = int(mm[1]), mm[2], STRAT[mm[3]], int(mm[4]), int(mm[5])
        z = np.load(f)
        ev, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"].astype(np.float64)
        assert np.array_equal(ev, eval_sets[fold])
        yc, tc = y[ctx], t[ctx]
        n0 = (tc == 0).sum()
        s0 = np.clip(yc[tc == 0].mean(), 0.5 / n0, 1 - 0.5 / n0)
        m = mu0[ev]
        p_s = expit(logit(m) + logit(s0) - logit(base0))
        key = (strategy, K, fold, seed)
        if key not in cache:
            cache[key] = risk_same_rows(x, y, ctx, ev)
        p_l = cache[key]
        lo, hi = p_s < 0.5, p_s > 0.5
        row = dict(ds=ds, model=model, strategy=strategy, K=K, fold=fold, seed=seed, ctx_pos=yc.mean(), s0=s0, ate_hat=s.mean(),
                   above=hi.mean(), rho_mu0=sp(s, m), rho_v_shift=sp(s, p_s * (1 - p_s)),
                   rho_lo=sp(s[lo], m[lo]), rho_hi=sp(s[hi], m[hi]),
                   rho_p_logit=np.nan if p_l is None else sp(s, p_l), rho_v_logit=np.nan if p_l is None else sp(s, p_l * (1 - p_l)))
        dec = np.minimum((np.argsort(np.argsort(m)) * NB) // len(m), NB - 1)
        for b in range(NB):
            row[f"tau{b}"], row[f"p{b}"] = s[dec == b].mean(), p_s[dec == b].mean()
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
df["sgn"] = np.sign(df.ate_hat)
for c in ["rho_mu0", "rho_v_shift", "rho_lo", "rho_hi", "rho_p_logit", "rho_v_logit"]:
    df["s_" + c] = df.sgn * df[c]
print("\nA. Alignment of the score with risk and with p(1-p), each multiplied by the sign of the predicted average effect.")
print("   A constant log-odds shift predicts: s_rho_v > 0 everywhere, s_rho_lo > 0, s_rho_hi < 0.")
for ds, d in df.groupby("ds", sort=False):
    g = d.groupby(["model", "strategy"]).agg(
        n=("K", "size"), ctx_pos=("ctx_pos", "mean"), above=("above", "mean"), ate_hat=("ate_hat", "mean"),
        rho_mu0=("rho_mu0", "mean"), s_rho_mu0=("s_rho_mu0", "mean"), s_rho_v_shift=("s_rho_v_shift", "mean"),
        s_rho_lo=("s_rho_lo", "mean"), s_rho_hi=("s_rho_hi", "mean"), s_rho_v_logit=("s_rho_v_logit", "mean"),
        v_pos=("s_rho_v_shift", lambda v: (v > 0).mean()), lo_pos=("s_rho_lo", lambda v: (v.dropna() > 0).mean()),
        hi_neg=("s_rho_hi", lambda v: (v.dropna() < 0).mean()), n_hi=("s_rho_hi", lambda v: v.notna().sum()))
    idx = [(m, s) for m in ["dopfn", "causalpfn", "tabpfn_t"] for s in ORDER if (m, s) in g.index]
    print(f"\n##### {ds}")
    print(g.reindex(idx).round(3).to_string())

print("\nB. Do-PFN, K=1024: mean tau-hat by decile of mu0-hat (low risk to high), and the context-scale risk p in those deciles")
e = df[(df.model == "dopfn") & (df.K == 1024)]
for ds, d in e.groupby("ds", sort=False):
    print(f"\n##### {ds}")
    tau = d.groupby("strategy")[[f"tau{b}" for b in range(NB)]].mean().reindex([s for s in ORDER if s in set(d.strategy)])
    pp = d.groupby("strategy")[[f"p{b}" for b in range(NB)]].mean().reindex(tau.index)
    print(tau.round(3).to_string())
    print(pp.round(3).to_string())
