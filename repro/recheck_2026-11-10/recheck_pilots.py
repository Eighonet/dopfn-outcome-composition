"""Re-score the saved CPU pilots of recoded outcome and treatment (Do-PFN, Criteo, K=1024). No model runs.

The pilot scripts printed rho against a risk model fitted on the first 150,000 holdout control rows; the Criteo file
is not shuffled, so that is one low-base-rate segment. Here every cell is scored against the risk model of the main
re-check (all holdout control rows). The two fold-0 files hold the first 2,048 evaluation rows, which are all
treated: no Qini there. The fold-3 file holds all 8,000 rows and is compared with the saved natural GPU runs.
"""
import os
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
H18 = REPRO / "home18"
os.environ["USERPROFILE"] = str(H18)
os.environ["HOME"] = str(H18)
sys.path.insert(0, str(REPRO / "home/projects/smiles/cpu_smoke"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklift.metrics import qini_auc_score

from fm_dense_cache import cached_dense_load

OUT = Path(sys.argv[1])
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
CV = H18 / "projects/smiles/results/fm_cv/criteo"
NAMES = {"T": "balanced_treatment", "R": "random"}
pd.set_option("display.width", 250)

x, t, y, _, _ = cached_dense_load(DATA_ROOT)
x = np.asarray(pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)
holdout = np.load(CV / "fold0_dopfn_balanced_treatment_ctx1024_seed42_eval8000_cuda0.predictions.npz")["final_holdout_indices"]
r = holdout[t[holdout] == 0]
m0 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(x[r], y[r])
print(f"mu0 fitted on {len(r)} holdout control rows", flush=True)

rows = []


def score(src, fold, seed, cell, readout, tau, mu, mu_pilot, yy, tt, natural=None, code1_lower=None):
    """One row per (file, cell, readout); tau is the effect of the actual treatment on the original outcome."""
    row = dict(file=src, fold=fold, seed=seed, cell=cell, readout=readout, n=len(tau), treated=int(tt.sum()), positives=int(yy.sum()),
               q=float(qini_auc_score(yy, tau, tt)) if 0 < tt.sum() < len(tt) else np.nan,
               rho=spearmanr(tau, mu).statistic, rho_pilot_mu0=spearmanr(tau, mu_pilot).statistic,
               mean=tau.mean(), median=np.median(tau), neg_share=(tau < 0).mean(),
               above1=(np.abs(tau) > 1).mean(), max_abs=np.abs(tau).max(), code1_lower=code1_lower)
    if natural is not None:
        row["spearman_nat"], row["pearson_nat"] = spearmanr(tau, natural).statistic, np.corrcoef(tau, natural)[0, 1]
    rows.append(row)


def code1_lower(a, cell, readout):
    """Share of rows where the arm coded 1 gets the lower prediction (original outcome labels)."""
    y1, y0 = a[f"{cell}|{readout}|y1"], a[f"{cell}|{readout}|y0"]
    return float(np.mean(y0 < y1) if "swap" in cell else np.mean(y1 < y0))


# fold 0, seed 42: first 2,048 evaluation rows (all treated)
for name in ("dopfn_norm_smoke_2048.npz", "dopfn_flip_smoke_2048.npz"):
    a = np.load(HERE / name)
    ev = a["ev"]
    mu = m0.predict_proba(x[ev])[:, 1]
    cells = sorted({k.split("|")[0] for k in a.files if "|" in k})
    for cell in cells:
        if name.startswith("dopfn_norm") and not cell.endswith(":nat"):
            continue                                                  # overridden target statistics: withdrawn, see README
        nat_cell = cell.split(":")[0] + ":nat"
        for readout in ("mean", "median"):
            tau = a[f"{cell}|{readout}|y1"] - a[f"{cell}|{readout}|y0"]
            nat = None
            for src in (a, np.load(HERE / "dopfn_norm_smoke_2048.npz")):
                if f"{nat_cell}|{readout}|y1" in src.files and nat_cell != cell:
                    nat = src[f"{nat_cell}|{readout}|y1"] - src[f"{nat_cell}|{readout}|y0"]
                    break
            score(name, 0, 42, cell, readout, tau, mu, a["mu0"], y[ev], t[ev], nat, code1_lower(a, cell, readout))

# fold 3, seed 44: all 8,000 evaluation rows, relabelled treatment against the saved natural GPU run
name = "dopfn_flip_smoke_fold3_seed44.npz"
a = np.load(HERE / name)
ev = a["ev"]
mu = m0.predict_proba(x[ev])[:, 1]
print(f"fold 3: Q(mu0) = {qini_auc_score(y[ev], mu, t[ev]):+.4f}; with the pilot's risk model {qini_auc_score(y[ev], a['mu0'], t[ev]):+.4f}")
for d in ("T", "R"):
    z = np.load(CV / f"fold3_dopfn_{NAMES[d]}_ctx1024_seed44_eval8000_cuda0.predictions.npz")
    assert np.array_equal(z["eval_indices"], ev)
    nat = z["predictions"].astype(np.float64)
    sw = a[f"{d}:swap|mean|y1"] - a[f"{d}:swap|mean|y0"]
    swm = a[f"{d}:swap|median|y1"] - a[f"{d}:swap|median|y0"]
    args = (mu, a["mu0"], y[ev], t[ev])
    score(name, 3, 44, f"{d}:nat (saved GPU run)", "mean", nat, *args, None, float((nat < 0).mean()))
    score(name, 3, 44, f"{d}:nat negated", "mean", -nat, *args, nat)
    score(name, 3, 44, f"{d}:swap", "mean", sw, *args, nat, code1_lower(a, f"{d}:swap", "mean"))
    score(name, 3, 44, f"{d}:swap", "median", swm, *args, nat, code1_lower(a, f"{d}:swap", "median"))
    score(name, 3, 44, f"{d}:(nat+swap)/2", "mean", (nat + sw) / 2, *args, nat)
    score(name, 3, 44, f"{d}:(nat-swap)/2", "mean", (nat - sw) / 2, *args, nat)

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.drop(columns=["file", "seed", "positives"]).round(4).to_string(index=False))

five = pd.read_csv(HERE / "recheck_five_h18.csv")
ref = five[(five.ds == "criteo") & (five.model == "dopfn") & (five.K == 1024) & (five.fold == 3) & (five.seed == 44) & five.strategy.isin(["T", "R"])]
print("\nsame contexts in recheck_five_h18.csv (Q, negated, same-rows risk ranking, Q(mu0), context contrast):")
print(ref[["strategy", "q", "q_neg", "risk_q", "mu0_q", "ctx_pos", "ctx_dm", "ate_hat"]].round(4).to_string(index=False))
