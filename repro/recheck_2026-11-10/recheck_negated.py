"""Q-hat of the negated score and of the score with a label-free sign fix, on the saved Criteo and LZD runs (CPU).

Also re-derives the headline numbers of the re-check from the three CSVs in recheck_2026-11-10/.
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
from sklift.metrics import qini_auc_score

import fm_benchmark as fb
from fm_dense_cache import cached_dense_load

OUT = Path(sys.argv[1])
HERE = REPRO / "recheck_2026-11-10"
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
RESULTS = H18 / "projects/smiles/results"
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}
MIRROR = {"R": 1, "T": 1, "10%": 1, "20%": 1, "35%": 1, "TY": 0, "75%": -1}
ORDER = ["R", "T", "10%", "20%", "35%", "TY", "75%"]
pd.set_option("display.width", 250)

# --- A: headline numbers from the existing CSVs
cv, ats, lzd = (pd.read_csv(HERE / n) for n in ("recheck_cv.csv", "recheck_ate_sign.csv", "recheck_lzd.csv"))
print("recheck_cv columns:", cv.columns.tolist())
for name, d in (("criteo", ats), ("lzd", lzd)):
    for model, m in d.groupby("model"):
        m = m[m.strategy != "TY"]
        ok = np.sign(m.rho) == np.sign(m.ate_hat) * m.strategy.map(MIRROR)
        print(f"A {name:6s} {model:10s} sign rule {ok.sum()}/{len(m)}  mean|rho|={m.rho.abs().mean():.2f}  "
              f"median |ate_hat|/|eval_dm|={(m.ate_hat.abs() / m.eval_dm.abs()).median():.1f}")

# --- B: negated score
rows = []
for ds in ("criteo", "lzd"):
    if ds == "criteo":
        _, t, y, _, _ = cached_dense_load(DATA_ROOT)
    else:
        _, t, y, _, _ = fb.load(DATA_ROOT, "lzd", use_group_hashes=False)
    t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)
    files = sorted(glob.glob(str(RESULTS / f"fm_cv/{ds}/fold*_eval8000_cuda0.predictions.npz")) +
                   glob.glob(str(RESULTS / f"fm_dose/{ds}/fold*_eval8000_cuda0.predictions.npz")))
    for f in files:
        mm = re.match(r"fold(\d+)_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx(\d+)_seed(\d+)_", Path(f).name)
        z = np.load(f); ev, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"].astype(np.float64)
        yy, tt = y[ev], t[ev]
        rows.append(dict(ds=ds, model=mm[2], strategy=STRAT[mm[3]], K=int(mm[4]), fold=int(mm[1]), seed=int(mm[5]),
                         q=float(qini_auc_score(yy, s, tt)), q_neg=float(qini_auc_score(yy, -s, tt)), ate_hat=s.mean(),
                         ctx_dm=y[ctx][t[ctx] == 1].mean() - y[ctx][t[ctx] == 0].mean()))
    print("scored", ds, len(files), flush=True)
df = pd.DataFrame(rows)
true_sign = 1.0                                                   # observed difference in means is positive on both datasets
df["q_ateflip"] = np.where(np.sign(df.ate_hat) == true_sign, df.q, df.q_neg)                 # flip when the predicted ATE has the wrong sign
df["q_rule"] = np.where(np.sign(df.ate_hat) * df.strategy.map(MIRROR) < 0, df.q_neg, df.q)   # make the score risk-aligned by the sign rule
df.to_csv(OUT, index=False)

key = ["model", "strategy", "K", "fold", "seed"]
base = {"criteo": cv, "lzd": lzd}
for ds in ("criteo", "lzd"):
    b = base[ds]
    col = [c for c in b.columns if c.startswith("risk_logit")][0]
    d = df[(df.ds == ds) & (df.K == 1024)].merge(b[key + [col]].rename(columns={col: "risk"}), on=key, how="left")
    for model in ("dopfn", "causalpfn", "tabpfn_t"):
        m = d[d.model == model]
        g = m.assign(win_raw=m.q > m.risk, win_flip=m.q_ateflip > m.risk, win_rule=m.q_rule > m.risk, flipped=m.q_ateflip != m.q) \
             .groupby("strategy")[["q", "q_neg", "q_ateflip", "q_rule", "risk", "flipped", "win_raw", "win_flip", "win_rule"]].mean()
        print(f"\nB {ds} {model} K=1024 (mean over 15 contexts)")
        print(g.reindex([s for s in ORDER if s in g.index]).round(3).to_string())
d = df[(df.model == "dopfn") & df.strategy.isin(["R", "T"])]
print("\nDo-PFN, R and T at every K: mean Q raw / negated / flip-when-predicted-ATE-is-negative, share of runs with Q>0")
print(d.assign(pos_raw=d.q > 0, pos_flip=d.q_ateflip > 0).groupby(["ds", "strategy", "K"])[["q", "q_neg", "q_ateflip", "pos_raw", "pos_flip"]].mean().round(3).to_string())
