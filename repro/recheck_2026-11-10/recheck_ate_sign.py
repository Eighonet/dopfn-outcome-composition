"""Does the sign of the predicted average effect explain the sign of the risk alignment? (CPU, saved predictions)

For every saved Criteo CV run: mean / median of tau-hat on the eval rows (the model's predicted ATE), share of
negative tau-hat, Spearman rho(tau-hat, mu0-hat), Q-hat, the context's own difference in means and the
difference in means on the eval rows. Second part: shape of the two per-arm predictions in the CPU pilot.
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklift.metrics import qini_auc_score

from fm_dense_cache import cached_dense_load

OUT, PILOT = Path(sys.argv[1]), Path(sys.argv[2])
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
RESULTS = H18 / "projects/smiles/results"
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}

x, t, y, _, _ = cached_dense_load(DATA_ROOT)
x = np.asarray(pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)

files = sorted(glob.glob(str(RESULTS / "fm_cv/criteo/fold*_eval8000_cuda0.predictions.npz")) +
               glob.glob(str(RESULTS / "fm_dose/criteo/fold*_eval8000_cuda0.predictions.npz")))
holdout = np.load(files[0])["final_holdout_indices"]
eval_all = np.unique(np.concatenate([np.load(f)["eval_indices"] for f in files]))
r = holdout[t[holdout] == 0]
m0 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(x[r], y[r])
mu0 = np.full(len(x), np.nan)
mu0[eval_all] = m0.predict_proba(x[eval_all])[:, 1]

rows = []
for f in files:
    mm = re.match(r"fold(\d+)_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx(\d+)_seed(\d+)_", Path(f).name)
    z = np.load(f); ev, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"].astype(np.float64)
    yc, tc = y[ctx], t[ctx]
    rows.append(dict(model=mm[2], strategy=STRAT[mm[3]], K=int(mm[4]), fold=int(mm[1]), seed=int(mm[5]),
                     ate_hat=s.mean(), ate_med=np.median(s), neg_share=(s < 0).mean(), sd=s.std(),
                     rho=spearmanr(s, mu0[ev]).statistic, qhat=float(qini_auc_score(y[ev], s, t[ev])),
                     ctx_dm=yc[tc == 1].mean() - yc[tc == 0].mean(),
                     eval_dm=y[ev][t[ev] == 1].mean() - y[ev][t[ev] == 0].mean()))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

pd.set_option("display.width", 250)
order = ["R", "T", "10%", "20%", "35%", "TY", "75%"]
k = df[df.K == 1024]
g = k.groupby(["model", "strategy"]).agg(n=("rho", "size"), ate_hat=("ate_hat", "mean"), ate_med=("ate_med", "mean"),
                                           neg_share=("neg_share", "mean"), rho=("rho", "mean"), qhat=("qhat", "mean"),
                                           ctx_dm=("ctx_dm", "mean"), eval_dm=("eval_dm", "mean"))
print(g.reindex([(m, s) for m in ["dopfn", "causalpfn", "tabpfn_t"] for s in order if (m, s) in g.index]).round(3).to_string())
for model, d in df.groupby("model"):
    same = (np.sign(d.ate_hat) == np.sign(d.rho)).mean()
    same_q = (np.sign(d.ate_hat) == np.sign(d.qhat)).mean()
    same_med = (np.sign(d.ate_med) == np.sign(d.rho)).mean()
    print(f"{model:10s} runs={len(d)} sign(ate_hat)==sign(rho): {same:.3f}  sign(ate_med)==sign(rho): {same_med:.3f}  "
          f"sign(ate_hat)==sign(Q): {same_q:.3f}  spearman(ate_hat, rho)={spearmanr(d.ate_hat, d.rho).statistic:+.2f}  "
          f"spearman(ate_hat, Q)={spearmanr(d.ate_hat, d.qhat).statistic:+.2f}  share ate_hat<0: {(d.ate_hat < 0).mean():.3f}  "
          f"share ctx_dm<0: {(d.ctx_dm < 0).mean():.3f}")
d = df[df.model == "dopfn"]
print("\nDo-PFN, by K and design: sign agreement of ate_hat with rho / with Q, and of ate_hat with the context's own difference in means")
print(d.assign(a=np.sign(d.ate_hat) == np.sign(d.rho), b=np.sign(d.ate_hat) == np.sign(d.qhat),
               c=np.sign(d.ate_hat) == np.sign(d.ctx_dm), neg=d.ate_hat < 0, qneg=d.qhat < 0)
      .groupby(["strategy", "K"])[["a", "b", "c", "neg", "qneg"]].mean().round(2).to_string())
# rule: "flag the run if the predicted ATE is negative while the context's difference in means is positive"
flag = (d.ate_hat < 0) & (d.ctx_dm > 0)
print(f"\nrule 'predicted ATE < 0 although context diff-in-means > 0': flagged {flag.sum()} of {len(d)} Do-PFN runs; "
      f"Q<0 among flagged {(d.qhat[flag] < 0).mean():.3f}; Q<0 among not flagged {(d.qhat[~flag] < 0).mean():.3f}")
neg = d.ate_hat < 0
print(f"rule 'predicted ATE < 0': flagged {neg.sum()}; Q<0 among flagged {(d.qhat[neg] < 0).mean():.3f}; "
      f"Q<0 among not flagged {(d.qhat[~neg] < 0).mean():.3f}; mean Q flagged {d.qhat[neg].mean():+.3f} / not {d.qhat[~neg].mean():+.3f}")

print("\nPilot (one context per design, 2,048 rows): shape of the per-arm predictions, readout = bucket mass above 0.5")
p = np.load(PILOT)
for cell in ["R:nat", "T:nat", "10%:nat", "20%:nat", "TY:nat"]:
    for k_ in ("ppos", "mean"):
        y1, y0 = p[f"{cell}|{k_}|y1"], p[f"{cell}|{k_}|y0"]
        slope = np.polyfit(y0, y1, 1)[0]
        lslope = np.polyfit(np.log(np.clip(y0, 1e-6, None)), np.log(np.clip(y1, 1e-6, None)), 1)[0]
        tau = y1 - y0
        print(f"{cell:8s} {k_:5s} pearson(y1,y0)={np.corrcoef(y1, y0)[0, 1]:+.3f} spearman={spearmanr(y1, y0).statistic:+.3f} "
              f"slope y1~y0={slope:.2f} log-log slope={lslope:.2f} ratio of means={y1.mean() / y0.mean():.2f} "
              f"sd(tau)/sd(y0)={tau.std() / y0.std():.2f} rho(tau,y0)={spearmanr(tau, y0).statistic:+.2f}")
