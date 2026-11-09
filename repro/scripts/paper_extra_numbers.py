"""Numbers quoted in the WSDM paper that no other project script writes, recomputed from saved results.

  python scripts/paper_extra_numbers.py      (run from repro/ with HOME/USERPROFILE = repro/home)

Writes rerun_compare/paper_extra_numbers.csv with one row per quantity:
  - full Criteo holdout, Do-PFN K=1,024, T and T+Y: uplift among the 10% top- and bottom-scored users, mean, median
    and share of negative predictions, per context seed; Welch intervals for T+Y - T (contexts treated as independent)
  - 200k holdout subsample: ranking by the baseline-risk estimate mu0 alone; mean and median prediction at 20%
  - CV: rho(score, mu0) for R and 75% on Criteo; LZD Q-hat by design and the 20% - T+Y and T+Y - T fold contrasts
"""
import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "home/projects/smiles/cpu_smoke"))

import numpy as np
import pandas as pd
from scipy import stats

from gpu_smoke import calculate_metrics

CMP = REPRO / "rerun_compare"
HOLD = REPRO / "home/projects/smiles/results/fm_holdout/criteo"
rows = []


def add(name, value, low=None, high=None, note=""):
    rows.append(dict(quantity=name, value=value, low=low, high=high, note=note))


def t_ci(v):
    v = np.asarray(v, float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half


def uplift_in(score, y, t, lo, hi):
    q = pd.Series(score).rank(pct=True).to_numpy()
    m = (q > lo) & (q <= hi)
    return 100 * (y[m & (t == 1)].mean() - y[m & (t == 0)].mean())


# full holdout, T and T+Y
metrics = {}
for strategy, arm in (("balanced_treatment", "T"), ("balanced_treatment_outcome", "TY")):
    for seed in (42, 43, 44):
        stem = HOLD / f"holdout_dopfn_{strategy}_ctx1024_seed{seed}_evalall_cuda0"
        z = np.load(f"{stem}.predictions.npz")
        p, y, t = z["predictions"].astype(float), z["y"], z["t"]
        m = json.loads(Path(f"{stem}.json").read_text())["metrics"]
        metrics.setdefault(arm, []).append((m["qini_auc_normalized"], m["auuc"], 100 * m["uplift@10%"]))
        add(f"full holdout {arm} seed {seed}: uplift top 10% (pp)", uplift_in(p, y, t, 0.9, 1.0))
        add(f"full holdout {arm} seed {seed}: uplift bottom 10% (pp)", uplift_in(p, y, t, 0.0, 0.1))
        add(f"full holdout {arm} seed {seed}: mean prediction (pp)", 100 * p.mean())
        add(f"full holdout {arm} seed {seed}: median prediction (pp)", 100 * np.median(p))
        add(f"full holdout {arm} seed {seed}: share of negative predictions", (p < 0).mean())
add("full holdout ATE (pp)", 100 * (y[t == 1].mean() - y[t == 0].mean()))
add("full holdout visit rate", y.mean())
for i, name in enumerate(("qhat", "auuc", "uplift@10% (pp)")):
    a, b = np.array(metrics["TY"])[:, i], np.array(metrics["T"])[:, i]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    d = a.mean() - b.mean()
    add(f"full holdout T+Y - T, {name}, Welch", d, d - half, d + half, f"df {df:.1f}")

# 200k subsample
cache = np.load(CMP / "holdout_share_cache.npz")
m, _ = calculate_metrics(cache["mu0"].astype(float), cache["y"], cache["t"])
add("subsample: ranking by mu0 alone, qhat", m["qini_auc_normalized"])
add("subsample: ranking by mu0 alone, auuc", m["auuc"])
add("subsample: ranking by mu0 alone, uplift@10% (pp)", 100 * m["uplift@10%"])
z10 = np.load(CMP / "hs10_scores.npz")
means = [100 * z10[f"20%_{s}"].mean() for s in range(42, 52)]
medians = [100 * np.median(z10[f"20%_{s}"]) for s in range(42, 52)]
add("subsample 20%: mean prediction over 10 contexts (pp)", np.mean(means), min(means), max(means), "low/high = range")
add("subsample 20%: median prediction (pp)", np.mean(medians), min(medians), max(medians), "low/high = range")

# CV
mech = pd.read_csv(CMP / "mechanism_runs.csv")
crit = mech[(mech.dataset == "criteo") & (mech.model == "dopfn")]
for s in ("R", "75%"):
    add(f"Criteo CV rho(score, mu0), {s}", crit[crit.strategy == s].groupby("fold").rho_mu0.mean().mean())
lzd = mech[(mech.dataset == "lzd") & (mech.model == "dopfn")]
for s, g in lzd.groupby("strategy"):
    add(f"LZD CV qhat, {s}", *t_ci(g.groupby("fold").qhat.mean()), note=f"rho {g.rho_mu0.mean():+.3f}")
pv = lzd.pivot_table(index="fold", columns="strategy", values="qhat")
add("LZD CV 20% - T+Y, qhat", *t_ci(pv["20%"] - pv["TY"]))
add("LZD CV T+Y - T, qhat", *t_ci(pv["TY"] - pv["T"]))

out = pd.DataFrame(rows)
out.to_csv(CMP / "paper_extra_numbers.csv", index=False)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
