"""Paired 20% vs T+Y check of Do-PFN on the fixed 200k Criteo holdout subsample, 10 context seeds (DataSphere T4 runs).

  collect <results dir>        read the fetched job outputs, check rows, write rerun_compare/hs10_scores.npz and
                               per-seed metrics with t-intervals over the 10 contexts (light, local)
  boot <part> <parts> <draws>  two-level bootstrap slice: resample context seeds and row clusters together
  report                       merge the bootstrap slices into rerun_compare/hs10_contrasts.csv
Run from repro/ with HOME/USERPROFILE = repro/home (metric code comes from the report's gpu_smoke.py).
"""
import glob
import json
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "home/projects/smiles/cpu_smoke"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist

from gpu_smoke import calculate_metrics

OUT = REPRO / "rerun_compare"
SCORES = OUT / "hs10_scores.npz"
CACHE = OUT / "holdout_share_cache.npz"
SEEDS = tuple(range(42, 52))
ARMS = {"share20": "20%", "balanced_treatment_outcome": "TY"}


def metrics(score, y, t):
    m, _ = calculate_metrics(score, y, t)
    return m["qini_auc_normalized"], 100 * m["uplift@10%"], m["auuc"]


def collect(results):
    cache = np.load(CACHE)
    ev, y, t = cache["ev"], cache["y"], cache["t"]
    scores, rows = {}, []
    for strategy, arm in ARMS.items():
        for seed in SEEDS:
            hits = glob.glob(str(Path(results) / "**" / f"holdout_dopfn_{strategy}_ctx1024_seed{seed}_eval200000_cuda0.predictions.npz"),
                             recursive=True)
            assert len(hits) == 1, f"{strategy} seed {seed}: {hits}"
            z = np.load(hits[0])
            assert np.array_equal(z["eval_indices"], ev) and np.array_equal(z["y"], y) and np.array_equal(z["t"], t)
            meta = json.load(open(hits[0].replace(".predictions.npz", ".json")))
            s = z["predictions"].astype(np.float64)
            scores[f"{arm}_{seed}"] = s
            q, u, a = metrics(s, y, t)
            rows.append(dict(arm=arm, seed=seed, qhat=q, up10=u, auuc=a, rho_mu0=spearmanr(s, cache["mu0"]).statistic,
                             median_cate=float(np.median(s)), device=meta.get("device"),
                             gpu=meta.get("gpu_name") or meta.get("device_name"), model_time_sec=meta.get("model_time_sec")))
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "hs10_runs.csv", index=False)
    np.savez_compressed(SCORES, y=y, t=t, cluster=cache["cluster"], tau_s=cache["tau_s"], tau_t=cache["tau_t"], **scores)
    wide = runs.pivot(index="seed", columns="arm")
    lines = []
    for m in ("qhat", "up10", "auuc", "rho_mu0"):
        d = (wide[m]["20%"] - wide[m]["TY"]).to_numpy()
        half = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        lines.append(dict(contrast="20% - TY", metric=m, mean=d.mean(), sd=d.std(ddof=1), seeds_positive=int((d > 0).sum()),
                          seeds=len(d), t_low=d.mean() - half, t_high=d.mean() + half))
    for arm in ("20%", "TY"):
        for ref in ("tau_s", "tau_t"):
            q_ref = metrics(cache[ref], y, t)[0]
            d = wide["qhat"][arm].to_numpy() - q_ref
            half = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            lines.append(dict(contrast=f"{arm} - ref {ref}", metric="qhat", mean=d.mean(), sd=d.std(ddof=1),
                              seeds_positive=int((d > 0).sum()), seeds=len(d), t_low=d.mean() - half, t_high=d.mean() + half))
    summary = pd.DataFrame(lines)
    summary.to_csv(OUT / "hs10_seed_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(runs.groupby("arm")[["qhat", "up10", "auuc", "rho_mu0", "median_cate"]].agg(["mean", "std"]).round(4).to_string())
    print(summary.round(4).to_string(index=False))


def boot(part, parts, draws):
    z = dict(np.load(SCORES))
    y, t, cl = z["y"], z["t"], z["cluster"]
    order = np.argsort(cl, kind="stable"); starts = np.searchsorted(cl[order], np.arange(cl.max() + 1))
    sizes = np.diff(np.append(starts, len(cl)))
    rng = np.random.default_rng([20261110, part])
    out = {"qhat": [], "up10": [], "q20_minus_refS": []}
    for _ in range(draws // parts):
        seeds = rng.choice(SEEDS, len(SEEDS), replace=True)
        picked = rng.integers(0, len(sizes), len(sizes))
        rows = np.concatenate([order[starts[i]:starts[i] + sizes[i]] for i in picked])
        yy, tt = y[rows], t[rows]
        cache = {}
        for s in set(seeds):
            for arm in ("20%", "TY"):
                cache[arm, s] = metrics(z[f"{arm}_{s}"][rows], yy, tt)
        ref = metrics(z["tau_s"][rows], yy, tt)[0]
        out["qhat"].append(np.mean([cache["20%", s][0] - cache["TY", s][0] for s in seeds]))
        out["up10"].append(np.mean([cache["20%", s][1] - cache["TY", s][1] for s in seeds]))
        out["q20_minus_refS"].append(np.mean([cache["20%", s][0] for s in seeds]) - ref)
    np.savez(OUT / f"hs10_boot_part{part}.npz", **{k: np.array(v) for k, v in out.items()})
    print(f"part {part}: {draws // parts} draws", flush=True)


def report():
    parts = sorted(OUT.glob("hs10_boot_part*.npz"))
    merged = {k: np.concatenate([np.load(p)[k] for p in parts]) for k in ("qhat", "up10", "q20_minus_refS")}
    rows = [dict(contrast=name, draws=len(v), mean=v.mean(), low=np.percentile(v, 2.5), high=np.percentile(v, 97.5),
                 share_le_0=float((v <= 0).mean())) for name, v in
            (("20% - TY, qhat", merged["qhat"]), ("20% - TY, up10 pp", merged["up10"]),
             ("20% - ref S-learner, qhat", merged["q20_minus_refS"]))]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hs10_contrasts.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    act = sys.argv[1]
    if act == "collect":
        collect(sys.argv[2])
    elif act == "boot":
        boot(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        report()
