"""Merge the bootstrap parts and print the summary in the same format as bootstrap_criteo_dopfn.py."""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

parts = int(sys.argv[1]) if len(sys.argv) > 1 else 10
analysis = Path.home() / "projects/smiles/results/analysis"
frames = [pd.read_csv(analysis / f"bootstrap_parallel_part{i}_of{parts}.csv") for i in range(parts)]
df = pd.concat(frames).sort_values("rep").reset_index(drop=True)
assert df.rep.tolist() == list(range(1, len(df) + 1)), "missing or duplicate replicates"
out = analysis / f"fm_criteo_dopfn_cluster_bootstrap_B{len(df)}_parallel.csv"
df.to_csv(out, index=False)
holdout = Path.home() / "projects/smiles/results/fm_holdout/criteo"
metrics = lambda st, s: json.load(open(holdout / f"holdout_dopfn_{st}_ctx1024_seed{s}_evalall_cuda0.json"))["metrics"]
names = {"qini": "qini_auc_normalized", "auuc": "auuc", "uplift10": "uplift@10%"}
print(f"Bootstrap {len(df)}/{len(df)}, merged from {parts} parts")
print("\n===== PAIRED CLUSTER BOOTSTRAP =====")
for metric, key in names.items():
    obs = np.mean([metrics("balanced_treatment_outcome", s)[key] - metrics("balanced_treatment", s)[key] for s in (42, 43, 44)])
    lo, hi = np.percentile(df[f"{metric}_delta_mean"].to_numpy(), [2.5, 97.5])
    print(f"{metric:9s} observed={obs:+.5f} bootstrap95=[{lo:+.5f}, {hi:+.5f}]")
print(f"Results: {out}")
print("FINAL: PASS")
