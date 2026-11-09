"""(b) Summarise the outcome-share dose-response of Do-PFN, K=1024, next to the existing T and T+Y runs.

Per share: Q-hat and uplift@10% averaged over seeds per fold, then mean and SD over the 5 folds.
T (balanced_treatment) is the natural share, T+Y (balanced_treatment_outcome) is 50%.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

H18 = Path(__file__).resolve().parents[1] / "home18/projects/smiles/results"
rows = []
for folder, label in (("fm_cv", None), ("fm_dose", None)):
    for path in glob.glob(str(H18 / folder / "*" / "fold*_dopfn_*_ctx1024_seed*_eval8000_cuda0.json")):
        r = json.load(open(path, encoding="utf8"))
        if r.get("status") != "ok" or r["strategy"] == "random":
            continue
        c = r["context_counts"]
        share = (c["T0Y1"] + c["T1Y1"]) / sum(c.values())
        rows.append(dict(dataset=r["dataset"], strategy=r["strategy"], fold=r["fold"], seed=r["model_seed"],
                         share=share, qhat=r["metrics"]["qini_auc_normalized"], up10=r["metrics"]["uplift@10%"]))
df = pd.DataFrame(rows)
fold = df.groupby(["dataset", "strategy", "fold"]).agg(share=("share", "mean"), qhat=("qhat", "mean"),
                                                      up10=("up10", "mean"), seeds=("seed", "nunique"))
out = fold.groupby(["dataset", "strategy"]).agg(share=("share", "mean"), q_mean=("qhat", "mean"), q_sd=("qhat", "std"),
                                               up10_mean=("up10", "mean"), up10_sd=("up10", "std"),
                                               folds=("qhat", "size"), q_pos_folds=("qhat", lambda v: int((v > 0).sum())))
out = out.reset_index().sort_values(["dataset", "share"])
out.to_csv(H18.parents[3] / "rerun_compare/novelty_b_dose.csv", index=False)
print(out.round(4).to_string(index=False))
