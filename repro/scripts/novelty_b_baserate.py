"""(b) Does the T+Y gain over T track the outcome base rate? Uses the reproduced CV JSONs only.

Base rate comes from the random-strategy context counts (the report's own outcome definition).
Q-hat is averaged over seeds per fold, then over folds, as in build_smiles_report.
LZD and Criteo come from home18 (sklearn 1.8 folds, the ones that match the report).
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPRO = Path(__file__).resolve().parents[1]
ROOTS = {"hillstrom": "home", "retailhero": "home", "orange": "home", "lzd": "home18", "criteo": "home18"}
STRATEGIES = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY"}
rows = []
for dataset, home in ROOTS.items():
    for path in glob.glob(str(REPRO / home / "projects/smiles/results/fm_cv" / dataset / "fold*_eval8000_cuda0.json")):
        r = json.load(open(path, encoding="utf8"))
        if not r.get("metrics"):
            continue
        c = r.get("context_counts") or {}
        rows.append(dict(dataset=dataset, model=r["model"], strategy=STRATEGIES[r["strategy"]], fold=r["fold"],
                         k=int(r.get("context_size") or Path(path).name.split("_ctx")[1].split("_")[0]),
                         seed=r["model_seed"], qhat=r["metrics"]["qini_auc_normalized"],
                         up10=r["metrics"]["uplift@10%"],
                         base=(c.get("T0Y1", 0) + c.get("T1Y1", 0)) / max(1, sum(c.values()))))
df = pd.DataFrame(rows)
base = df[df.strategy == "R"].groupby("dataset").base.mean()
cell = (df.groupby(["dataset", "model", "k", "strategy", "fold"])[["qhat", "up10"]].mean()
          .groupby(["dataset", "model", "k", "strategy"]).mean().unstack("strategy"))
out = pd.DataFrame({"base_rate": cell.index.get_level_values("dataset").map(base),
                    "q_R": cell["qhat"]["R"], "q_T": cell["qhat"]["T"], "q_TY": cell["qhat"]["TY"],
                    "dq_TY_T": cell["qhat"]["TY"] - cell["qhat"]["T"],
                    "dup10_TY_T": cell["up10"]["TY"] - cell["up10"]["T"]}, index=cell.index).reset_index()
out.to_csv(REPRO / "rerun_compare/novelty_b_baserate.csv", index=False)
print(f"{len(df)} CV runs; base rates:", base.round(4).to_dict())
print(out.round(4).to_string(index=False))
for model, g in out.groupby("model"):
    per = g.groupby("dataset")[["base_rate", "dq_TY_T"]].mean()
    print(f"{model}: Spearman(base rate, mean dQ T+Y-T over K) = {spearmanr(per.base_rate, per.dq_TY_T).statistic:+.2f} over {len(per)} datasets")
print(f"all cells: Spearman = {spearmanr(out.base_rate, out.dq_TY_T).statistic:+.2f} over {len(out)} cells; "
      f"T inverted (q_T<0) in {(out.q_T < 0).sum()} cells, of which T+Y fixes sign in {((out.q_T < 0) & (out.q_TY > 0)).sum()}")
