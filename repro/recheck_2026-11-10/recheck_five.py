"""Sign rule, negated score and same-rows risk ranking on the saved CV runs of all five datasets.

Reads the `home/` tree by default: designs R, T, T+Y, four context sizes, three models, 540 runs per dataset
(for Criteo and LZD these are the scikit-learn 1.5.2 folds, i.e. a second fold split). CPU only.
A third argument `home18` reads the paper's folds instead (Criteo and LZD), with the share designs of `fm_dose`.
"""
import glob
import json
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
from fm_dense_cache import cached_dense_load

OUT = Path(sys.argv[1])
SETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["hillstrom", "retailhero", "orange", "lzd", "criteo"]
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
TREE = sys.argv[3] if len(sys.argv) > 3 else "home"
SUBS = ["fm_cv", "fm_dose"] if TREE == "home18" else ["fm_cv"]
RESULTS = REPRO / TREE / "projects/smiles/results"
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}
NB = 10
rng = np.random.default_rng(0)


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


def q(yy, s, tt):
    return float(qini_auc_score(yy, s, tt))


def decomp(s, m, yy, tt):
    r0 = rankdata(m + 1e-12 * rng.standard_normal(len(m)), method="ordinal")
    b = np.minimum((r0 - 1) * NB // len(m), NB - 1)
    df = pd.DataFrame(dict(s=s, b=b))
    between = df.groupby("b").s.transform("mean").to_numpy()
    within = df.groupby("b").s.rank(pct=True).to_numpy()
    return dict(q_between=q(yy, between, tt), q_within=q(yy, within, tt))


def risk_same_rows(x, y, ctx, ev):
    """Risk-only ranking fitted on the context rows; the treatment column is not used."""
    xc, yc = x[ctx], y[ctx]
    if len(np.unique(yc)) < 2:
        return None
    with np.errstate(all="ignore"):
        med = np.nanmedian(xc, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda a: np.where(np.isfinite(a), a, med)
    mu, sd = fill(xc).mean(0), fill(xc).std(0) + 1e-9
    return LogisticRegression(C=1.0, max_iter=2000).fit((fill(xc) - mu) / sd, yc).predict_proba((fill(x[ev]) - mu) / sd)[:, 1]


rows = []
for ds in SETS:
    if ds == "criteo":
        x, t, y, _, _ = cached_dense_load(DATA_ROOT)
        x = np.asarray(pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
    else:
        X, t, y, _, _ = fb.load(DATA_ROOT, ds, use_group_hashes=False)
        x = numeric(X)
    t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)
    files = [f for sub in SUBS for f in sorted(glob.glob(str(RESULTS / sub / ds / "fold*_eval8000_cuda0.predictions.npz")))]
    holdout = np.load(files[0])["final_holdout_indices"]
    eval_sets = {}
    for f in files:
        fold = int(re.match(r"fold(\d+)_", Path(f).name)[1])
        if fold not in eval_sets:
            eval_sets[fold] = np.load(f)["eval_indices"]
    eval_all = np.unique(np.concatenate(list(eval_sets.values())))
    assert not np.intersect1d(holdout, eval_all).size
    # mu0: control rows never evaluated; for the grouped datasets only the group-disjoint holdout
    pool = holdout if ds in ("criteo", "lzd") else np.setdiff1d(np.arange(len(y)), eval_all)
    r = pool[t[pool] == 0]
    m0 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(x[r], y[r])
    mu0 = np.full(len(y), np.nan)
    mu0[eval_all] = m0.predict_proba(x[eval_all])[:, 1]
    print(f"\n=== {ds}: rows {len(y)}, features {x.shape[1]}, treated share {t.mean():.3f}, "
          f"y|t=0 {y[t == 0].mean():.4f}, y|t=1 {y[t == 1].mean():.4f}, diff {y[t == 1].mean() - y[t == 0].mean():+.4f}; "
          f"mu0 on {len(r)} control rows; eval sets {len(eval_sets)}", flush=True)

    cache, dq, da = {}, 0.0, 0.0
    for f in files:
        mm = re.match(r"fold(\d+)_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx(\d+)_seed(\d+)_", Path(f).name)
        fold, model, strategy, K, seed = int(mm[1]), mm[2], STRAT[mm[3]], int(mm[4]), int(mm[5])
        z = np.load(f)
        ev, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"].astype(np.float64)
        yy, tt = y[ev], t[ev]
        assert np.array_equal(yy, z["y"]) and np.array_equal(tt, z["t"]) and np.array_equal(ev, eval_sets[fold])
        rec = json.load(open(f.replace(".predictions.npz", ".json")))
        yc, tc = y[ctx], t[ctx]
        row = dict(ds=ds, model=model, strategy=strategy, K=K, fold=fold, seed=seed, ctx_pos=yc.mean(),
                   pos_t0=yc[tc == 0].mean(), pos_t1=yc[tc == 1].mean(),
                   q=q(yy, s, tt), q_neg=q(yy, -s, tt), rho=spearmanr(s, mu0[ev]).statistic, ate_hat=s.mean(),
                   neg_share=(s < 0).mean(), ctx_dm=yc[tc == 1].mean() - yc[tc == 0].mean(),
                   eval_dm=yy[tt == 1].mean() - yy[tt == 0].mean(), mu0_q=q(yy, mu0[ev], tt))
        dq = max(dq, abs(row["q"] - rec["metrics"]["qini_auc_normalized"]))
        da = max(da, abs(row["ate_hat"] - rec["score_mean"]))
        row.update(decomp(s, mu0[ev], yy, tt))
        key = (strategy, K, fold, seed)
        if key not in cache:
            sc = risk_same_rows(x, y, ctx, ev)
            cache[key] = (ctx.copy(), np.nan if sc is None else q(yy, sc, tt),
                          np.nan if sc is None else spearmanr(sc, mu0[ev]).statistic)
        assert np.array_equal(cache[key][0], ctx), "contexts differ between models"
        row["risk_q"], row["risk_rho"] = cache[key][1], cache[key][2]
        rows.append(row)
    print(f"max |Q - recorded Q| = {dq:.2e}; max |mean score - recorded| = {da:.2e}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
df["mirror"] = np.sign(0.5 - df.ctx_pos)
df["rule_ok"] = np.sign(df.rho) == np.sign(df.ate_hat) * df.mirror
df["gap"] = df.q - df.risk_q
df["gap_neg"] = df.q_neg - df.risk_q
for ds, d in df.groupby("ds", sort=False):
    print(f"\n##### {ds}")
    g = d.groupby(["model", "strategy"]).agg(n=("q", "size"), ctx_pos=("ctx_pos", "mean"), q=("q", "mean"), q_neg=("q_neg", "mean"),
                                             rho=("rho", "mean"), ate_hat=("ate_hat", "mean"), eval_dm=("eval_dm", "mean"),
                                             q_between=("q_between", "mean"), q_within=("q_within", "mean"),
                                             risk_q=("risk_q", "mean"), mu0_q=("mu0_q", "mean"), q_neg_share=("q", lambda v: (v < 0).mean()))
    print(g.round(3).to_string())
    nt = d[d.strategy != "TY"]
    print("sign rule, all designs but T+Y (sign rho = sign mean tau-hat x sign(0.5 - context share of positives)):")
    for model, e in nt.groupby("model"):
        print(f"  {model:10s} {int(e.rule_ok.sum())}/{len(e)}  mean |rho| {e.rho.abs().mean():.2f}  pearson(rho, Q) {np.corrcoef(e.rho, e.q)[0, 1]:+.2f}  "
              f"share with mean tau-hat of the sign of the observed effect {(np.sign(e.ate_hat) == np.sign(e.eval_dm.mean())).mean():.2f}  "
              f"median |mean tau-hat| / |observed| {np.median(np.abs(e.ate_hat)) / abs(e.eval_dm.mean()):.1f}")
    e = d[d.model == "dopfn"]
    fm = e.groupby(["strategy", "fold"])[["q", "q_neg", "risk_q", "gap", "gap_neg"]].mean()
    print("Do-PFN, fold means over K and seeds: Q, Q of the negated score, same-rows risk ranking")
    print(fm.round(3).unstack("fold").to_string())
    print("eval-noise floor: SD of Q(mu0) over the five eval sets =", round(d.groupby("fold").mu0_q.first().std(), 3),
          "| SD of Q(risk ranking) over folds =", round(d.groupby("fold").risk_q.mean().std(), 3))
