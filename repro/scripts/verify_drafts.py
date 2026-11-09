"""Check the numeric claims of the two student drafts against the reproduced holdout and CV results.

Holdout: repro/home/projects/smiles/results/fm_holdout (48 runs). CV: repro/home (Hillstrom, RetailHero,
Orange) and repro/home18 (LZD, Criteo; sklearn >= 1.8 folds, the ones that match the report).
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPRO = Path(__file__).resolve().parents[1]
HOLD = REPRO / "home/projects/smiles/results/fm_holdout"
CV_HOME = {"hillstrom": "home", "retailhero": "home", "orange": "home", "lzd": "home18", "criteo": "home18"}
SHORT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "B"}
M = ["qini_auc_normalized", "auuc", "uplift@5%", "uplift@10%", "uplift@20%", "uplift@30%"]

rows = []
for path in glob.glob(str(HOLD / "*" / "holdout_*_evalall_cuda0.json")):
    r = json.load(open(path))
    rows.append(dict(dataset=r["dataset"], model=r["model"], k=r["context_size"], strategy=r["strategy"],
                     seed=r["model_seed"], n_holdout=r["n_final_holdout"], **{m: r["metrics"][m] for m in M}))
hold = pd.DataFrame(rows)
print("holdout runs:", len(hold))
print("\nTable 1 (draft B) holdout rows:", hold.groupby("dataset").n_holdout.first().to_dict())

cfg = hold.groupby(["dataset", "model", "k", "strategy"]).agg(q_mean=("qini_auc_normalized", "mean"),
                                                                q_sd=("qini_auc_normalized", "std"),
                                                                up10=("uplift@10%", "mean"), seeds=("seed", "nunique"))
print("\nTable 3 (draft B) holdout Qini mean +- SD per configuration:")
print(cfg.round(5).to_string())

# CV mean Qini of the same configurations (seed mean per fold, then fold mean)
def cv_mean(dataset, model, k, strategy):
    files = glob.glob(str(REPRO / CV_HOME[dataset] / "projects/smiles/results/fm_cv" / dataset /
                          f"fold*_{model}_{strategy}_ctx{k}_seed*_eval8000_cuda0.json"))
    d = pd.DataFrame([dict(fold=json.load(open(f))["fold"], q=json.load(open(f))["metrics"]["qini_auc_normalized"])
                      for f in files])
    return d.groupby("fold").q.mean().mean(), len(d)

sel = cfg.reset_index()
sel = sel[~((sel.dataset == "criteo") & (sel.model == "dopfn") & (sel.strategy == "balanced_treatment"))]
sel[["cv_q", "cv_runs"]] = [cv_mean(*r) for r in sel[["dataset", "model", "k", "strategy"]].itertuples(index=False)]
sel["below_cv"] = sel.q_mean < sel.cv_q
sel["sign_change"] = np.sign(sel.q_mean) != np.sign(sel.cv_q)
print(f"\nselected configs: {len(sel)}; holdout below CV: {int(sel.below_cv.sum())}; sign changes: {int(sel.sign_change.sum())}")
print(sel[["dataset", "model", "k", "strategy", "cv_q", "q_mean", "below_cv", "sign_change"]].round(5).to_string(index=False))
lead = sel.groupby("dataset").apply(lambda g: pd.Series(dict(cv_leader=g.loc[g.cv_q.idxmax(), "model"],
                                                             hold_leader=g.loc[g.q_mean.idxmax(), "model"])))
print("\nleaders (CV vs holdout):"); print(lead.to_string()); print("leader changes:", int((lead.cv_leader != lead.hold_leader).sum()))

h = hold[(hold.dataset == "hillstrom")].groupby(["model", "k", "strategy"])["uplift@10%"].mean() * 100
print("\nHillstrom holdout uplift@10% (pp):", h.round(3).to_dict())

c = hold[(hold.dataset == "criteo") & (hold.model == "dopfn") & (hold.k == 1024)].pivot(index="seed", columns="strategy", values=M)
diff = pd.DataFrame({m: c[m]["balanced_treatment_outcome"] - c[m]["balanced_treatment"] for m in M})
print("\nDo-PFN Criteo K=1024, T+Y minus T per seed (uplift in pp):")
show = diff.copy(); show[[m for m in M if m.startswith("uplift")]] *= 100
print(show.round(4).to_string())
t975 = 4.302653  # t quantile, 2 df
for m in M:
    v = show[m].to_numpy(); se = v.std(ddof=1) / np.sqrt(3)
    print(f"  {m:20s} mean {v.mean():+.4f}  three-seed t-interval [{v.mean() - t975 * se:+.4f}, {v.mean() + t975 * se:+.4f}]")
