"""Are the predicted effects feasible for a 0/1 outcome? Share of evaluation rows with |tau-hat| above 1 and above 0.5.

Saved CV predictions only (both result trees and the share designs); no data, no GPU.
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPRO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1])
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}
ORDER = ["R", "T", "10%", "20%", "35%", "TY", "75%"]

rows = []
for tree, sub in [("home", "fm_cv"), ("home18", "fm_cv"), ("home18", "fm_dose")]:
    for f in glob.glob(str(REPRO / tree / "projects/smiles/results" / sub / "*" / "fold*_eval8000_cuda0.predictions.npz")):
        mm = re.match(r"fold(\d+)_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx(\d+)_seed(\d+)_", Path(f).name)
        s = np.load(f)["predictions"].astype(np.float64)
        a = np.abs(s)
        rows.append(dict(tree=tree, ds=Path(f).parent.name, model=mm[2], strategy=STRAT[mm[3]], K=int(mm[4]), fold=int(mm[1]), seed=int(mm[5]),
                         above1=(a > 1).mean(), above05=(a > 0.5).mean(), max_abs=a.max(), q99_abs=np.quantile(a, 0.99),
                         mean=s.mean(), median=np.median(s), neg_share=(s < 0).mean()))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
g = df.groupby(["ds", "tree", "model", "strategy"]).agg(
    n=("K", "size"), above1=("above1", "mean"), runs_above1=("above1", lambda v: (v > 0).mean()), above05=("above05", "mean"),
    max_abs=("max_abs", "max"), q99_abs=("q99_abs", "mean"), mean=("mean", "mean"), median=("median", "mean"), neg_share=("neg_share", "mean"))
idx = [(a, b, m, s) for a in ["criteo", "lzd", "hillstrom", "retailhero", "orange"] for b in ["home18", "home"]
       for m in ["dopfn", "causalpfn", "tabpfn_t"] for s in ORDER if (a, b, m, s) in g.index]
print(g.reindex(idx).round(4).to_string())
print("\nDo-PFN, Criteo, natural composition, by K (both trees)")
e = df[(df.ds == "criteo") & (df.model == "dopfn") & df.strategy.isin(["R", "T"])]
print(e.groupby(["strategy", "K"]).agg(n=("K", "size"), above1=("above1", "mean"), above05=("above05", "mean"), max_abs=("max_abs", "max"),
                                       mean=("mean", "mean"), median=("median", "mean")).round(4).to_string())
