"""Analysis for repro/holdout_share_protocol.md: Do-PFN outcome shares on the fixed 200k Criteo holdout subsample.

  python holdout_share_analysis.py prepare   eval rows, T and T+Y scores, development-only reference learners
  python holdout_share_analysis.py analyze   add share runs, metrics per seed, t-intervals, paired cluster bootstrap
Run with HOME/USERPROFILE = repro/home. Writes repro/rerun_compare/holdout_share_*.csv and a cache .npz.
"""
import glob
import json
import os
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
CODE = REPRO / "home/projects/smiles/cpu_smoke"
sys.path.insert(0, str(CODE))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist
from sklearn.ensemble import HistGradientBoostingClassifier

import fm_benchmark as fm
from fm_dense_cache import cached_dense_load
from gpu_smoke import calculate_metrics

FROZEN = REPRO / "home/projects/smiles/results/fm_holdout/criteo"
SHARE = REPRO / "home/projects/smiles/results/fm_holdout_share/criteo"
CACHE = REPRO / "rerun_compare/holdout_share_cache.npz"
OUT = REPRO / "rerun_compare"
SEEDS = (42, 43, 44)
EXPECTED_EVAL_HASH = "cf7d585ab7d1bb46ef6714eab3758e4232b7034bd0dc68812de1b39cf4288bdb"
NAMES = {"balanced_treatment": "T", "balanced_treatment_outcome": "TY", "share20": "20%", "share35": "35%"}


def prepare():
    x, t, y, _, _ = cached_dense_load((REPRO / "home/projects/smiles/data/data_A_cleaned").resolve())
    frozen = {}
    for strategy in ("balanced_treatment", "balanced_treatment_outcome"):
        for seed in SEEDS:
            z = np.load(FROZEN / f"holdout_dopfn_{strategy}_ctx1024_seed{seed}_evalall_cuda0.predictions.npz")
            frozen[strategy, seed] = (z["eval_indices"], z["predictions"])
    final_idx = frozen["balanced_treatment", 42][0]
    ev = fm.choose_eval(final_idx, t, y, 200000, fold=0)
    assert fm.hash_indices(ev) == EXPECTED_EVAL_HASH, "eval subsample differs from the runs"
    scores = {}
    for (strategy, seed), (idx, pred) in frozen.items():
        pos = pd.Series(np.arange(len(idx)), index=idx)
        scores[f"{NAMES[strategy]}_{seed}"] = pred[pos.loc[ev].to_numpy()]
    # reference learners on development rows only (2M random rows), never on holdout rows
    dev = np.setdiff1d(np.arange(len(x)), final_idx, assume_unique=True)
    fit = np.sort(np.random.default_rng(20261109).choice(dev, 2_000_000, replace=False))
    hgb = lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0)
    m0 = hgb().fit(x[fit][t[fit] == 0], y[fit][t[fit] == 0])
    m1 = hgb().fit(x[fit][t[fit] == 1], y[fit][t[fit] == 1])
    s = hgb().fit(np.column_stack([x[fit], t[fit]]), y[fit])
    mu0, mu1 = m0.predict_proba(x[ev])[:, 1], m1.predict_proba(x[ev])[:, 1]
    tau_s = (s.predict_proba(np.column_stack([x[ev], np.ones(len(ev))]))[:, 1] -
             s.predict_proba(np.column_stack([x[ev], np.zeros(len(ev))]))[:, 1])
    _, cluster = np.unique(x[ev], axis=0, return_inverse=True)
    np.savez_compressed(CACHE, ev=ev, y=y[ev], t=t[ev], mu0=mu0, mu1=mu1, tau_t=mu1 - mu0, tau_s=tau_s,
                        cluster=cluster.ravel(), **{f"score_{k}": v for k, v in scores.items()})
    print(f"prepared: {len(ev):,} eval rows, {cluster.max() + 1:,} feature-vector clusters, "
          f"ATE {y[ev][t[ev] == 1].mean() - y[ev][t[ev] == 0].mean():+.4f}")


def metric_row(score, yy, tt):
    m, _ = calculate_metrics(score, yy, tt)
    return {"qhat": m["qini_auc_normalized"], "auuc": m["auuc"], "up10": 100 * m["uplift@10%"]}


def analyze(draws=1000):
    c = dict(np.load(CACHE))
    ev, yy, tt = c["ev"], c["y"], c["t"]
    for strategy in ("share20", "share35"):
        for seed in SEEDS:
            f = SHARE / f"holdout_dopfn_{strategy}_ctx1024_seed{seed}_eval200000_cuda0.predictions.npz"
            if f.exists():
                z = np.load(f)
                assert np.array_equal(z["eval_indices"], ev) and np.array_equal(z["y"], yy)
                c[f"score_{NAMES[strategy]}_{seed}"] = z["predictions"]
    keys = sorted(k[6:] for k in c if k.startswith("score_"))
    rows = []
    for k in keys:
        s = c[f"score_{k}"]
        rows.append(dict(strategy=k.rsplit("_", 1)[0], seed=int(k.rsplit("_", 1)[1]), **metric_row(s, yy, tt),
                         rho_mu0=spearmanr(s, c["mu0"]).statistic, rho_tau_t=spearmanr(s, c["tau_t"]).statistic,
                         rho_tau_s=spearmanr(s, c["tau_s"]).statistic, mean_cate=float(s.mean()), sd_cate=float(s.std())))
    for name in ("tau_t", "tau_s"):
        rows.append(dict(strategy=f"ref_{name}", seed=0, **metric_row(c[name], yy, tt),
                         rho_mu0=spearmanr(c[name], c["mu0"]).statistic))
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "holdout_share_runs.csv", index=False)
    pd.set_option("display.width", 200)
    print(runs.round(4).to_string(index=False))
    have = set(runs.strategy)
    contrasts = [(a, b) for a, b in (("20%", "TY"), ("20%", "T"), ("35%", "20%"), ("TY", "T")) if {a, b} <= have]
    # seed-level summary and three-seed t-intervals
    summary = []
    for a, b in contrasts:
        for m in ("qhat", "auuc", "up10"):
            va = runs[runs.strategy == a].set_index("seed")[m]; vb = runs[runs.strategy == b].set_index("seed")[m]
            d = (va - vb).dropna().to_numpy()
            half = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            summary.append(dict(contrast=f"{a} - {b}", metric=m, mean=d.mean(), seeds_positive=int((d > 0).sum()),
                                seeds=len(d), t_low=d.mean() - half, t_high=d.mean() + half))
    boot = bootstrap(c, keys, contrasts, draws, np.random.default_rng(20261109)) if draws else {}
    for part in sorted(OUT.glob("holdout_share_boot_part*.npz")) if not draws else []:
        z = np.load(part)
        for name in z.files:
            a, b, m = name.split("|")
            boot.setdefault((a, b, m), []).extend(z[name].tolist())
    for row in summary:
        a, b = row["contrast"].split(" - ")
        if (a, b, row["metric"]) in boot:
            row["boot_draws"] = len(boot[a, b, row["metric"]])
            row["boot_low"], row["boot_high"] = np.percentile(boot[a, b, row["metric"]], [2.5, 97.5])
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "holdout_share_contrasts.csv", index=False)
    print(summ.round(4).to_string(index=False))


def bootstrap(c, keys, contrasts, draws, rng):
    """Paired cluster bootstrap of the seed-averaged difference; one draw serves every strategy and seed."""
    yy, tt, cl = c["y"], c["t"], c["cluster"]
    order = np.argsort(cl, kind="stable"); starts = np.searchsorted(cl[order], np.arange(cl.max() + 1))
    sizes = np.diff(np.append(starts, len(cl)))
    used = {s for pair in contrasts for s in pair}
    boot = {(a, b, m): [] for a, b in contrasts for m in ("qhat", "up10")}
    for _ in range(draws):
        picked = rng.integers(0, len(sizes), len(sizes))
        rows_ = np.concatenate([order[starts[i]:starts[i] + sizes[i]] for i in picked])
        cache = {k: metric_row(c[f"score_{k}"][rows_], yy[rows_], tt[rows_]) for k in keys if k.rsplit("_", 1)[0] in used}
        for a, b in contrasts:
            for m in ("qhat", "up10"):
                boot[a, b, m].append(np.mean([cache[f"{a}_{s}"][m] - cache[f"{b}_{s}"][m] for s in SEEDS]))
    return boot


def boot_part(part, parts, total):
    """One of `parts` independent slices of the bootstrap, each with its own generator; report merges them."""
    c = dict(np.load(CACHE))
    for strategy in ("share20", "share35"):
        for seed in SEEDS:
            z = np.load(SHARE / f"holdout_dopfn_{strategy}_ctx1024_seed{seed}_eval200000_cuda0.predictions.npz")
            c[f"score_{NAMES[strategy]}_{seed}"] = z["predictions"]
    keys = sorted(k[6:] for k in c if k.startswith("score_"))
    contrasts = [("20%", "TY"), ("20%", "T"), ("35%", "20%"), ("TY", "T")]
    boot = bootstrap(c, keys, contrasts, total // parts, np.random.default_rng([20261109, part]))
    np.savez(OUT / f"holdout_share_boot_part{part}.npz", **{f"{a}|{b}|{m}": np.array(v) for (a, b, m), v in boot.items()})
    print(f"part {part}: {total // parts} draws")


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "prepare":
        prepare()
    elif action == "boot":
        boot_part(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:   # analyze [draws]; with 0 draws it merges the saved bootstrap parts
        analyze(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
