"""Criteo final holdout: does the same-rows risk reference depend on the risk model, and does it hold at the top
of the ranking? CPU only, no foundation model. Post hoc and descriptive: nothing is selected on the holdout.

Usage: recheck_risk_holdout.py OUT_PREFIX [DRAWS [WORKERS]]
Reads the 38 saved holdout contexts (the file list of recheck_holdout.py). Every variant is fitted on the saved
context rows and scored on the rows the model was scored on (the full holdout or the fixed 200k-row subsample).
Risk-only rankings (the treatment column is not used):
  lr                 logistic regression, median fill, standardised, C=1: the reference line of recheck_five.py
  lr_C0.01, lr_C100  the same with another C
  hgb                HistGB (lr 0.05, 200 iterations) on the raw features
  lr_ctrl, lr_trt    the reference fitted on the control / treated rows of the context only
Uplift score from the same rows, for contrast:
  lr_tdiff           lr_trt - lr_ctrl, a logistic T-learner
References fitted on 2M development rows (Table 1 of the paper): mu0 (HistGB on the control rows, refitted here and
compared with the cached values), tau_s and tau_t (cached S- and T-learner).
Metrics as in the paper (sklift): normalized Qini and uplift@k with strategy "overall", in percentage points.
Writes OUT_PREFIX.csv (one row per context and score) and OUT_PREFIX_boot.npz (the bootstrap draws). The two-level
bootstrap resamples the ten context seeds and the feature-vector row clusters together, as scripts/hs10_analysis.py
does, with integer row weights instead of copies of the rows.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "6")
import glob
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
H18 = REPRO / "home18"
os.environ["USERPROFILE"] = str(H18)
os.environ["HOME"] = str(H18)
sys.path.insert(0, str(REPRO / "home/projects/smiles/cpu_smoke"))

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklift.metrics import qini_auc_score, uplift_at_k

DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
CACHE = REPRO / "rerun_compare/holdout_share_cache.npz"
HS10 = REPRO / "rerun_compare/hs10_scores.npz"
STRAT = {"balanced_treatment": "T", "balanced_treatment_outcome": "TY", "share20": "20%", "share35": "35%"}
KS = (5, 10, 20, 30)
METRICS = ["q"] + [f"up{k}" for k in KS]
RISK = ["lr", "lr_C0.01", "lr_C100", "hgb", "lr_ctrl", "lr_trt"]
SCORES = ["fm", "fm_neg"] + RISK + ["lr_tdiff", "mu0"]
BOOT = ["fm", "lr", "hgb", "lr_ctrl"]
SEEDS = tuple(range(42, 52))
ARM = {"TY": "TY", "20%": "S20"}


def metrics(yy, s, tt):
    if s is None:
        return dict.fromkeys(METRICS, float("nan"))
    out = {"q": float(qini_auc_score(y_true=yy, uplift=s, treatment=tt))}
    for k in KS:
        out[f"up{k}"] = 100 * float(uplift_at_k(y_true=yy, uplift=s, treatment=tt, strategy="overall", k=k / 100))
    return out


def prep(xc):
    """Median fill and standardisation fitted on the given rows, as in recheck_five.py."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(xc, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    fill = lambda a: np.where(np.isfinite(a), a, med)
    mu, sd = fill(xc).mean(0), fill(xc).std(0) + 1e-9
    return lambda a: (fill(a) - mu) / sd


def lr(xc, yc, xe, C=1.0):
    if len(np.unique(yc)) < 2:
        return None
    f = prep(xc)
    return LogisticRegression(C=C, max_iter=2000).fit(f(xc), yc).predict_proba(f(xe))[:, 1]


def hgb(xc, yc, xe):
    if len(np.unique(yc)) < 2:
        return None
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(xc, yc).predict_proba(xe)[:, 1]


def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return (v.mean(), tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (np.nan, np.nan)


class Ranked:
    """A score on the fixed rows, sorted once the way sklift sorts it (stable, reversed)."""

    def __init__(self, s):
        o = np.argsort(s, kind="mergesort")[::-1]
        self.o = o.astype(np.int32)
        self.thr = np.r_[np.where(np.diff(s[o]))[0], len(s) - 1].astype(np.int32)

    def cum(self, W):
        """Cumulative weighted counts down the ranking: rows, treated rows, treated responders, control responders."""
        return [np.cumsum(a[self.o]) for a in W]

    def qini_area(self, cums):
        """Area under the Qini curve of sklift.metrics.qini_curve and under its diagonal."""
        n_all, n_t, y_t, y_c = (a[self.thr] for a in cums)
        n_c = n_all - n_t
        curve = y_t - y_c * np.divide(n_t, n_c, out=np.zeros_like(n_t), where=n_c != 0)
        if curve[0] != 0 or n_all[0] != 0:
            n_all, curve = np.r_[0.0, n_all], np.r_[0.0, curve]
        return float(np.trapz(curve, n_all)), float(n_all[-1] * curve[-1] / 2)


def row_weights(w, y, t):
    return (w, w * t, w * y * t, w * y * (1 - t))


def weighted_metrics(r, W, perfect, k=0.1):
    """Normalized Qini and uplift@k (overall, pp) that sklift returns on the rows replicated W[0] times."""
    cums = r.cum(W)
    area = r.qini_area(cums)[0]
    n_top = int(cums[0][-1] * k)
    j = int(np.searchsorted(cums[0], n_top, side="right"))        # the rows [0, j) of the ranking are taken whole
    top = [a[j - 1] if j else 0.0 for a in cums]
    rest = n_top - top[0]
    if rest > 0:                                                  # a part of the copies of row j
        oj = r.o[j]
        top = [v + rest * a[oj] / W[0][oj] for v, a in zip(top, W)]
    up = top[2] / top[1] - top[3] / (top[0] - top[1])
    return (area - perfect[1]) / (perfect[0] - perfect[1]), 100 * up


def check_weighted(scores, y, t, cluster):
    """The weighted metrics against sklift: on the rows as they are, and on rows replicated by bootstrap weights."""
    names = list(scores)
    ranked = {k: Ranked(scores[k]) for k in names}
    perfect = Ranked(y * t - y * (1 - t))
    yi, ti = y.astype(int), t.astype(int)
    W = row_weights(np.ones(len(y)), y, t)
    pa = perfect.qini_area(perfect.cum(W))
    d = [np.abs(np.array(weighted_metrics(ranked[k], W, pa)) -
                np.array([qini_auc_score(yi, scores[k], ti), 100 * uplift_at_k(yi, scores[k], ti, strategy="overall", k=0.1)]))
         for k in names]
    print(f"weighted metrics against sklift, {len(names)} scores, unit weights: max abs difference "
          f"Q-hat {max(v[0] for v in d):.1e}, uplift@10% {max(v[1] for v in d):.1e} pp")
    rng = np.random.default_rng(1)
    n_cl = int(cluster.max()) + 1
    worst = np.zeros(2)
    for _ in range(2):
        w = np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl)[cluster].astype(np.float64)
        W = row_weights(w, y, t)
        pa = perfect.qini_area(perfect.cum(W))
        rep = np.repeat(np.arange(len(y)), w.astype(int))
        for k in names[::max(1, len(names) // 9)]:
            ref = [qini_auc_score(yi[rep], scores[k][rep], ti[rep]),
                   100 * uplift_at_k(yi[rep], scores[k][rep], ti[rep], strategy="overall", k=0.1)]
            worst = np.maximum(worst, np.abs(np.array(weighted_metrics(ranked[k], W, pa)) - np.array(ref)))
    print(f"the same on rows replicated by two bootstrap weight vectors: Q-hat {worst[0]:.1e}, uplift@10% {worst[1]:.1e} pp")
    assert max(v[0] for v in d) < 1e-9 and max(v[1] for v in d) < 1e-9 and worst.max() < 1e-9


def boot_part(job):
    path, part, draws = job
    with np.load(path) as z:
        y, t, cluster = z["y"].astype(np.float64), z["t"].astype(np.float64), z["cluster"]
        names = [k for k in z.files if k not in ("y", "t", "cluster")]
        ranked = [Ranked(z[k]) for k in names]
    perfect = Ranked(y * t - y * (1 - t))
    n_cl = int(cluster.max()) + 1
    rng = np.random.default_rng([20261110, 3, part])
    vals, counts = np.empty((draws, len(names), 2)), np.empty((draws, len(SEEDS)))
    for d in range(draws):
        counts[d] = np.bincount(rng.integers(0, len(SEEDS), len(SEEDS)), minlength=len(SEEDS))
        w = np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl)[cluster].astype(np.float64)
        W = row_weights(w, y, t)
        pa = perfect.qini_area(perfect.cum(W))
        for i, r in enumerate(ranked):
            vals[d, i] = weighted_metrics(r, W, pa)
    return names, vals, counts


def main():
    out = Path(sys.argv[1])
    draws = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    from fm_dense_cache import cached_dense_load
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)

    x, t, y, _, _ = cached_dense_load(DATA_ROOT)
    x = np.asarray(pd.DataFrame(x).apply(pd.to_numeric, errors="coerce"), dtype=np.float64)
    t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)
    c = np.load(CACHE)
    ev, cluster = c["ev"], c["cluster"]
    assert np.array_equal(y[ev], c["y"]) and np.array_equal(t[ev], c["t"])
    holdout = np.load(sorted(glob.glob(str(REPRO / "home/projects/smiles/results/fm_cv/criteo/fold0_*.npz")))[0])["final_holdout_indices"]
    assert np.isin(ev, holdout).all()

    # Table 1's risk reference: HistGB on the control rows of the same 2M development rows
    dev = np.setdiff1d(np.arange(len(y)), holdout, assume_unique=True)
    fit = np.sort(np.random.default_rng(20261109).choice(dev, 2_000_000, replace=False))
    r = fit[t[fit] == 0]
    m0 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(x[r], y[r])
    mu0 = np.full(len(y), np.nan)
    mu0[holdout] = m0.predict_proba(x[holdout])[:, 1]
    print(f"criteo: rows {len(y)}, features {x.shape[1]}; holdout {len(holdout)}; 200k subsample {len(ev)} rows in "
          f"{cluster.max() + 1} feature-vector clusters; mu0 refitted on {len(r)} control rows of the 2M development rows; "
          f"max |refit - cached mu0| on the 200k rows = {np.abs(mu0[ev] - c['mu0']).max():.1e}", flush=True)
    for label, idx in (("full holdout", holdout), ("200k subsample", ev)):
        print(f"  {label}: treated share {t[idx].mean():.4f}, response rate {100 * y[idx].mean():.2f}%, "
              f"average effect {100 * (y[idx][t[idx] == 1].mean() - y[idx][t[idx] == 0].mean()):.2f} pp")

    files = (sorted(glob.glob(str(REPRO / "home/projects/smiles/results/fm_holdout/criteo/*.npz"))) +
             sorted(glob.glob(str(REPRO / "home/projects/smiles/results/fm_holdout_share/criteo/*.npz"))) +
             sorted(glob.glob(str(REPRO / "ds/results/hs_*/unpacked/repro/home18/projects/smiles/results/fm_holdout_share/criteo/*.npz"))))
    rows, keep, last = [], {}, (None, None)
    for f in files:
        mm = re.match(r"holdout_(causalpfn|dopfn|tabpfn_t)_(.+)_ctx(\d+)_seed(\d+)_eval(all|\d+)_", Path(f).name)
        model, strategy, K, seed, size = mm[1], STRAT[mm[2]], int(mm[3]), int(mm[4]), mm[5]
        source = "ds" if "hs_" in f else "home"
        z = np.load(f)
        idx, ctx, s = z["eval_indices"], z["context_indices"], z["predictions"].astype(np.float64)
        assert np.array_equal(y[idx], z["y"]) and np.array_equal(t[idx], z["t"]) and np.isin(idx, holdout).all()
        assert not np.isin(ctx, holdout).any() and len(ctx) == K
        assert np.array_equal(idx, ev) if size != "all" else len(idx) == len(holdout)
        if last[0] is None or not np.array_equal(last[0], idx):
            last = (idx, x[idx])
        xe, xc, yc, tc = last[1], x[ctx], y[ctx], t[ctx]
        c0, c1 = tc == 0, tc == 1
        sc = {"fm": s, "fm_neg": -s, "lr": lr(xc, yc, xe), "lr_C0.01": lr(xc, yc, xe, C=0.01),
              "lr_C100": lr(xc, yc, xe, C=100.0), "hgb": hgb(xc, yc, xe),
              "lr_ctrl": lr(xc[c0], yc[c0], xe), "lr_trt": lr(xc[c1], yc[c1], xe)}
        sc["lr_tdiff"] = None if sc["lr_ctrl"] is None or sc["lr_trt"] is None else sc["lr_trt"] - sc["lr_ctrl"]
        sc["mu0"] = mu0[idx]
        parts = [(source, idx, slice(None))]
        if size == "all" and model == "dopfn":       # the same scores on the 200k rows, as Table 1 does for T
            parts.append((source + "_sub", ev, pd.Series(np.arange(len(idx)), index=idx).loc[ev].to_numpy()))
        for src, ridx, sel in parts:
            yy, tt = y[ridx], t[ridx]
            for name in SCORES:
                v = sc[name]
                rows.append(dict(source=src, model=model, strategy=strategy, K=K, seed=seed, n_eval=len(ridx),
                                 ctx_pos=int(yc.sum()), ctx_ctrl=int(c0.sum()), ctx_pos_ctrl=int(yc[c0].sum()),
                                 score=name, **metrics(yy, None if v is None else v[sel], tt)))
        if source == "ds":
            for name in BOOT:
                keep[f"{ARM[strategy]}_{seed}_{name}"] = sc[name]
        done = rows[-len(SCORES) * len(parts)]
        print(f"{source:5s} {model:9s} {strategy:4s} K={K} seed={seed} eval={size}: Q-hat {done['q']:+.4f}, "
              f"lr {rows[-len(SCORES) * len(parts) + 2]['q']:+.4f}", flush=True)
    yy, tt = c["y"].astype(int), c["t"].astype(int)
    for name in ("mu0", "tau_s", "tau_t"):
        rows.append(dict(source="ref", model="histgb_2M", strategy=name, K=2_000_000, seed=0, n_eval=len(ev),
                         score="fm", **metrics(yy, c[name], tt)))
    df = pd.DataFrame(rows)
    df.to_csv(out.with_suffix(".csv"), index=False)

    key = ["n_eval", "source", "model", "strategy", "K"]
    g = df.groupby(key + ["score"], sort=False)[METRICS].mean()
    n = df[df.score == "fm"].groupby(key, sort=False).size().rename("n")
    print("\n##### Check against Table 1 of the paper (Q-hat / uplift@10% pp): 200k rows: T (3) -0.053 / 1.43, 35% (3) -0.007 / 2.12, "
          "T+Y (10) 0.049 / 3.50, 20% (10) 0.074 / 4.67, S-learner 0.082 / 5.36, T-learner 0.079 / 5.28, mu0 0.081 / 4.34; "
          "full holdout: T (3) -0.046 / 1.76, T+Y (3) 0.048 / 3.56")
    chk = g.xs("fm", level="score")[["q", "up10"]].join(n)
    print(chk[chk.index.get_level_values("model").isin(["dopfn", "histgb_2M"])].round(4).to_string())

    for m, title in (("q", "Q-hat"), ("up10", "uplift@10%, pp"), ("up5", "uplift@5%, pp"), ("up20", "uplift@20%, pp"),
                     ("up30", "uplift@30%, pp")):
        print(f"\n##### {title}: mean over the contexts of each group, by score")
        print(g[m].unstack("score")[SCORES].join(n).drop(index="ref", level="source").round(4 if m == "q" else 2).to_string())

    print("\n##### Seed-paired gap to each reference, mean [95% t-interval over contexts], contexts below the reference")
    piv = {m: df.pivot_table(index=key + ["seed"], columns="score", values=m, sort=False) for m in ("q", "up10")}
    for target in ("fm", "fm_neg"):
        for m, digits in (("q", 3), ("up10", 2)):
            res = {}
            for grp, e in piv[m].groupby(key, sort=False):
                if grp[1] == "ref" or (target == "fm_neg" and grp[3] not in ("T", "35%")):
                    continue
                for ref in RISK + ["lr_tdiff", "mu0"]:
                    d = (e[target] - e[ref]).dropna()
                    mean, half = ci(d)
                    res.setdefault(" ".join(map(str, grp)), {})[ref] = (
                        f"{mean:+.{digits}f} [{mean - half:+.{digits}f}, {mean + half:+.{digits}f}] {int((d < 0).sum())}/{len(d)}")
            print(f"\n{'model score' if target == 'fm' else 'negated model score'} minus reference, "
                  f"{'Q-hat' if m == 'q' else 'uplift@10% pp'}")
            print(pd.DataFrame(res).T.to_string())

    # two-level bootstrap on the 200k rows: ten context seeds x feature-vector row clusters
    hs = np.load(HS10)
    assert np.array_equal(hs["y"], c["y"]) and np.array_equal(hs["t"], c["t"]) and np.array_equal(hs["cluster"], cluster)
    print("\n##### Two-level bootstrap on the 200k rows")
    print("max |saved run score - hs10_scores.npz|:",
          max(float(np.abs(keep[f"{ARM[a]}_{s}_fm"] - hs[f"{a}_{s}"]).max()) for a in ARM for s in SEEDS))
    for name in ("mu0", "tau_s", "tau_t"):
        keep[f"ref_{name}"] = c[name].astype(np.float64)
    assert all(v is not None for v in keep.values())
    yf, tf = c["y"].astype(np.float64), c["t"].astype(np.float64)
    check_weighted(keep, yf, tf, cluster)
    with tempfile.TemporaryDirectory() as tmp:               # the scores reach the worker processes through a file
        path = str(Path(tmp) / "scores.npz")
        np.savez(path, y=c["y"], t=c["t"], cluster=cluster, **keep)
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(boot_part, [(path, p, draws // workers) for p in range(workers)]))
    names, vals, counts = res[0][0], np.concatenate([v[1] for v in res]), np.concatenate([v[2] for v in res])
    np.savez_compressed(str(out) + "_boot.npz", names=np.array(names), vals=vals, seed_counts=counts)
    col = {k: i for i, k in enumerate(names)}
    W1, perfect = row_weights(np.ones(len(yf)), yf, tf), Ranked(yf * tf - yf * (1 - tf))
    area1 = perfect.qini_area(perfect.cum(W1))
    point = {k: weighted_metrics(Ranked(keep[k]), W1, area1) for k in names}

    def series(name, m):
        """Draws and point value of a series: a fixed reference, or the mean over the (resampled) context seeds."""
        if name.startswith("ref_"):
            return vals[:, col[name], m], point[name][m]
        arm, score = name.split(".")
        v = np.stack([vals[:, col[f"{arm}_{s}_{score}"], m] for s in SEEDS], axis=1)
        return (v * counts).sum(1) / counts.sum(1), float(np.mean([point[f"{arm}_{s}_{score}"][m] for s in SEEDS]))

    contrasts = [("20% Do-PFN - T+Y Do-PFN (paper: Q-hat [0.005, 0.043], uplift@10% [-0.1, 2.4])", "S20.fm", "TY.fm"),
                 ("20% Do-PFN - S-learner 2M (paper: Q-hat [-0.022, +0.006])", "S20.fm", "ref_tau_s"),
                 ("T+Y Do-PFN - risk LR, same rows", "TY.fm", "TY.lr"),
                 ("T+Y Do-PFN - risk HistGB, same rows", "TY.fm", "TY.hgb"),
                 ("T+Y Do-PFN - risk LR, control rows of the context", "TY.fm", "TY.lr_ctrl"),
                 ("T+Y Do-PFN - mu0 2M", "TY.fm", "ref_mu0"),
                 ("20% Do-PFN - risk LR, same rows", "S20.fm", "S20.lr"),
                 ("20% Do-PFN - risk HistGB, same rows", "S20.fm", "S20.hgb"),
                 ("20% Do-PFN - risk LR, control rows of the context", "S20.fm", "S20.lr_ctrl"),
                 ("20% Do-PFN - mu0 2M", "S20.fm", "ref_mu0"),
                 ("risk LR on T+Y rows - mu0 2M", "TY.lr", "ref_mu0"),
                 ("risk LR on 20% rows - mu0 2M", "S20.lr", "ref_mu0"),
                 ("risk LR on T+Y rows - risk LR on 20% rows", "TY.lr", "S20.lr"),
                 ("S-learner 2M - mu0 2M", "ref_tau_s", "ref_mu0"),
                 ("T-learner 2M - mu0 2M", "ref_tau_t", "ref_mu0"),
                 ("S-learner 2M - risk LR on T+Y rows", "ref_tau_s", "TY.lr"),
                 ("S-learner 2M - risk LR on 20% rows", "ref_tau_s", "S20.lr")]
    print(f"{len(vals)} draws ({workers} streams); point value, bootstrap mean [2.5%, 97.5%], share of draws <= 0")
    for m, title, digits in ((0, "Q-hat", 4), (1, "uplift@10%, pp", 2)):
        print(f"\n{title}")
        for label, a, b in contrasts:
            (da, pa), (db, pb) = series(a, m), series(b, m)
            d = da - db
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f"  {label:78s} {pa - pb:+.{digits}f}  {d.mean():+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]  {np.mean(d <= 0):.3f}")
    print("\nLevels in the bootstrap (point value, [2.5%, 97.5%])")
    for name in ("TY.fm", "S20.fm", "TY.lr", "S20.lr", "TY.hgb", "S20.hgb", "TY.lr_ctrl", "S20.lr_ctrl", "ref_mu0", "ref_tau_s", "ref_tau_t"):
        parts = []
        for m, digits in ((0, 4), (1, 2)):
            d, p = series(name, m)
            lo, hi = np.percentile(d, [2.5, 97.5])
            parts.append(f"{p:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]")
        print(f"  {name:12s} Q-hat {parts[0]}   uplift@10% {parts[1]} pp")


if __name__ == "__main__":
    main()
