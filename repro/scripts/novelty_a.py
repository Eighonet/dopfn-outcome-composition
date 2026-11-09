"""(a) How much of the CausalPFN ranking is the plain neighbour mean? Run from repro/rerun.

Usage: novelty_a.py <predictions.csv> <label> <fm column>:<local column> [...] [--draws N]
For each pair: Spearman rho of FM vs local mean; metrics of FM, local mean, local mean with FM only
breaking its ties, and the FM residual after a linear fit on local-mean ranks. Residual and tie-break
metrics get an arm-stratified bootstrap interval (same resampling as the E16 runner, own seed).
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, "tools")
import run_m01_e16_v001 as e16

p = argparse.ArgumentParser()
p.add_argument("predictions")
p.add_argument("label")
p.add_argument("pairs", nargs="+")
p.add_argument("--draws", type=int, default=200)
a = p.parse_args()
df = pd.read_csv(a.predictions)
y, t, ids = df.Y_benefit.to_numpy(), df["T"].to_numpy(), df.source_row.to_numpy()
arms = [np.flatnonzero(t == arm) for arm in (0, 1)]


def metrics(score, rows=slice(None)):
    row = e16.metric_row("x", score[rows], y[rows], t[rows], ids[rows])
    return row["normalized_qini_auc"], row["uplift_10_pp"]


def interval(score):
    rng = np.random.default_rng(20261107)
    values = []
    for _ in range(a.draws):
        rows = np.concatenate([rng.choice(arm, len(arm), replace=True) for arm in arms])
        values.append(metrics(score, rows))
    return np.percentile(np.array(values), [2.5, 97.5], axis=0)


out = []
for pair in a.pairs:
    fm_col, local_col = pair.split(":")
    fm, local = df[fm_col].to_numpy(float), df[local_col].to_numpy(float)
    rf, rl = rankdata(fm) / len(fm), rankdata(local) / len(local)
    resid = rf - np.polyval(np.polyfit(rl, rf, 1), rl)
    tiebreak = rl + 1e-9 * rf          # local-mean order, FM only orders rows tied on the local mean
    rho = float(spearmanr(fm, local).statistic)
    ties = 1 - len(np.unique(local)) / len(local)
    for name, score, ci in (("fm", fm, False), ("local_mean", local, False),
                            ("local_mean_fm_tiebreak", tiebreak, True), ("fm_residual", resid, True)):
        q, u = metrics(score)
        row = dict(data=a.label, pair=pair, score=name, spearman_fm_local=rho, local_tied_share=ties,
                   qhat=q, uplift10_pp=u)
        if ci:
            (q_lo, u_lo), (q_hi, u_hi) = interval(score)
            row.update(qhat_low=q_lo, qhat_high=q_hi, uplift10_low=u_lo, uplift10_high=u_hi)
        out.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
pd.DataFrame(out).to_csv(f"../rerun_compare/novelty_a_{a.label}.csv", index=False)
