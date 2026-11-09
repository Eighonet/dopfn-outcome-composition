"""(b) Mechanism probe on the saved CV runs: why does Do-PFN rank in reverse on rare-outcome data?

For Criteo and LZD at K=1024, every saved CV run (5 folds x 3 seeds; CausalPFN, Do-PFN, TabPFN-T under
R, T, T+Y, plus the Do-PFN outcome-share runs) is compared on its own 8000 eval rows with
  - mu0(x), mu1(x), tau(x) = mu1 - mu0 from a HistGB T-learner fitted on rows disjoint from every CV eval
    row (Criteo: the 2.09M-row final holdout; LZD: all rows whose feature vector never occurs among the
    CV eval rows), reported as Spearman correlations of the FM score with baseline risk and with uplift;
  - the same big-data T-learner scored on the same eval rows (a same-row classical reference);
  - budget-matched classical controls fitted on the run's own 1024 context rows (logistic T-learner and
    HistGB S-learner), to see whether the inversion is a property of the context or of Do-PFN.
Outputs repro/rerun_compare/mechanism_runs.csv and prints aggregated tables.
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklift.metrics import qini_auc_score

import fm_benchmark as fm
from fm_dense_cache import cached_dense_load

DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
RESULTS = H18 / "projects/smiles/results"
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}


def load(dataset):
    if dataset == "criteo":
        x, t, y, _, _ = cached_dense_load(DATA_ROOT)
    else:
        x, t, y, _, _ = fm.load(DATA_ROOT, dataset, use_group_hashes=False)
    x = np.asarray(pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
    return x, np.asarray(t).astype(int), np.asarray(y).astype(int)


def hgb():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0)


def qhat(y, s, t):
    return float(qini_auc_score(y, s, t))


def small_controls(x, t, y, ctx, ev):
    xc, tc, yc = x[ctx], t[ctx], y[ctx]
    med = np.nanmedian(xc, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda a: np.where(np.isfinite(a), a, med)
    mu, sd = fill(xc).mean(0), fill(xc).std(0) + 1e-9
    zc, ze = (fill(xc) - mu) / sd, (fill(x[ev]) - mu) / sd
    out = {}
    probs = []
    for arm in (1, 0):
        m = tc == arm
        if len(np.unique(yc[m])) < 2:
            probs.append(np.full(len(ev), yc[m].mean())); continue
        probs.append(LogisticRegression(C=1.0, max_iter=2000).fit(zc[m], yc[m]).predict_proba(ze)[:, 1])
    out["logit_T"] = probs[0] - probs[1]
    s = HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=20, random_state=0)
    s.fit(np.column_stack([xc, tc]), yc)
    out["hgb_S"] = (s.predict_proba(np.column_stack([x[ev], np.ones(len(ev))]))[:, 1] -
                    s.predict_proba(np.column_stack([x[ev], np.zeros(len(ev))]))[:, 1])
    return out


rows = []
for dataset in ("criteo", "lzd"):
    files = sorted(glob.glob(str(RESULTS / "fm_cv" / dataset / "fold*_ctx1024_seed*_eval8000_cuda0.predictions.npz")) +
                   glob.glob(str(RESULTS / "fm_dose" / dataset / "fold*_ctx1024_seed*_eval8000_cuda0.predictions.npz")))
    x, t, y = load(dataset)
    first = np.load(files[0])
    holdout = first["final_holdout_indices"]
    eval_all = np.unique(np.concatenate([np.load(f)["eval_indices"] for f in files[:200]]))
    if dataset == "criteo":
        fit_rows = holdout
    else:  # rows whose feature vector never occurs among CV eval rows (LZD has duplicated vectors)
        h = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
        fit_rows = np.flatnonzero(~np.isin(h, h[eval_all]))
    assert not np.intersect1d(fit_rows, eval_all).size
    models = {}
    for arm in (0, 1):
        r = fit_rows[t[fit_rows] == arm]
        models[arm] = hgb().fit(x[r], y[r])
    mu0 = np.full(len(x), np.nan); mu1 = np.full(len(x), np.nan)
    mu0[eval_all] = models[0].predict_proba(x[eval_all])[:, 1]
    mu1[eval_all] = models[1].predict_proba(x[eval_all])[:, 1]
    tau = mu1 - mu0
    print(f"{dataset}: reference T-learner fitted on {len(fit_rows):,} disjoint rows; eval rows {len(eval_all):,}", flush=True)
    controls = {}
    for f in files:
        m = re.match(r"fold(\d+)_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx\d+_seed(\d+)_", Path(f).name)
        fold, model, strategy, seed = int(m[1]), m[2], m[3], int(m[4])
        z = np.load(f); ev, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"]
        yy, tt = y[ev], t[ev]
        assert np.array_equal(yy, z["y"]) and np.array_equal(tt, z["t"]), "row alignment failed"
        row = dict(dataset=dataset, model=model, strategy=STRAT[strategy], fold=fold, seed=seed,
                   ctx_y_share_t0=float(y[ctx][t[ctx] == 0].mean()), ctx_y_share_t1=float(y[ctx][t[ctx] == 1].mean()),
                   qhat=qhat(yy, s, tt), rho_mu0=spearmanr(s, mu0[ev]).statistic, rho_mu1=spearmanr(s, mu1[ev]).statistic,
                   rho_tau=spearmanr(s, tau[ev]).statistic, score_mean=float(s.mean()), score_sd=float(s.std()),
                   ref_tlearner_qhat=qhat(yy, tau[ev], tt))
        key = (strategy, fold, seed)
        if key not in controls:
            controls[key] = {k: qhat(yy, v, tt) for k, v in small_controls(x, t, y, ctx, ev).items()}
        row.update({f"ctx_{k}_qhat": v for k, v in controls[key].items()})
        rows.append(row)
    print(f"{dataset}: {len(files)} runs probed", flush=True)

df = pd.DataFrame(rows)
df.to_csv(REPRO / "rerun_compare/mechanism_runs.csv", index=False)
agg = (df.groupby(["dataset", "model", "strategy"])
         [["qhat", "rho_mu0", "rho_tau", "ref_tlearner_qhat", "ctx_logit_T_qhat", "ctx_hgb_S_qhat", "score_mean"]]
         .mean().round(4))
order = {"R": 0, "T": 1, "10%": 2, "20%": 3, "35%": 4, "TY": 5, "75%": 6}
agg = agg.reset_index().assign(o=lambda d: d.strategy.map(order)).sort_values(["dataset", "model", "o"]).drop(columns="o")
pd.set_option("display.width", 200)
print(agg.to_string(index=False))
