"""Recompute the numbers quoted in paper/main.tex from saved results and compare them with the printed values.

CPU only, a few seconds. It reads run JSONs and analysis CSVs that are in the repository, never the prediction
arrays (*.npz). Numbers that other scripts derived from those arrays are read from their CSV outputs and are
marked "(from npz, via ...)" in the source column. Writes rerun_compare/paper_numbers.csv and exits with
status 1 if any value differs from the paper by more than its printed rounding.

  python repro/scripts/paper_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPRO = Path(__file__).resolve().parents[1]
ROOT = REPRO.parent
CMP = REPRO / "rerun_compare"
HOLD = REPRO / "home/projects/smiles/results/fm_holdout/criteo"
CV18 = REPRO / "home18/projects/smiles/results/fm_cv/criteo"
DSRUNS = REPRO / "ds/results"
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY"}
rows = []


def check(section, quantity, value, paper, decimals, source):
    """Record one number. paper is the printed value, or None for a number that is only reported."""
    ok = None if paper is None else bool(abs(value - paper) <= 0.5 * 10 ** -decimals + 1e-9)
    rows.append(dict(section=section, quantity=quantity, value=value, paper=paper, ok=ok, source=source))


def tci(x):
    """Mean and 95% t-interval."""
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def check_ci(section, quantity, x, paper, decimals, source):
    for part, value, printed in zip(("mean", "low", "high"), tci(x), paper):
        check(section, f"{quantity} [{part}]", value, printed, decimals, source)


# ---- Frozen full holdout, K=1,024 (Table 1, abstract, Setup) --------------------------------------------------
def holdout(model, strategy, k, seed):
    d = json.loads((HOLD / f"holdout_{model}_{strategy}_ctx{k}_seed{seed}_evalall_cuda0.json").read_text())
    m = d["metrics"]
    return dict(q=m["qini_auc_normalized"], auuc=m["auuc"], up10=100 * m["uplift@10%"], score=100 * d["score_mean"], meta=d)


SEEDS = (42, 43, 44)
SRC = "home/projects/smiles/results/fm_holdout/criteo/*.json"
T = [holdout("dopfn", "balanced_treatment", 1024, s) for s in SEEDS]
TY = [holdout("dopfn", "balanced_treatment_outcome", 1024, s) for s in SEEDS]
meta = T[0]["meta"]
check("Setup", "Criteo rows (millions)", meta["n_rows"] / 1e6, 13.98, 2, SRC)
check("Setup", "holdout rows", meta["n_final_holdout"], 2093918, 0, SRC)
check("Setup", "development rows (millions; 'almost 12M')", meta["n_development"] / 1e6, None, 0, SRC)
check("Setup", "features used (12 minus f7, f10)", meta["n_features"], 10, 0, SRC)
for name, runs, printed in (("T", T, (-0.046, -0.018, 1.76)), ("T+Y", TY, (0.048, 0.019, 3.56))):
    for key, value, dec in zip(("q", "auuc", "up10"), printed, (3, 3, 2)):
        check("Table 1, full holdout", f"{name}: {key}", np.mean([r[key] for r in runs]), value, dec, SRC)
for key, printed, dec in (("q", (0.094, 0.037, 0.150), 3), ("auuc", (0.037, 0.015, 0.059), 3),
                          ("up10", (1.80, -1.28, 4.88), 2)):
    check_ci("Table 1, full holdout", f"T+Y - T: {key}, paired by seed", [b[key] - a[key] for a, b in zip(T, TY)],
             printed, dec, SRC)
check("RQ1", "T: top-10% uplift, lowest seed (pp)", min(r["up10"] for r in T), 1.3, 1, SRC)
check("RQ1", "T: top-10% uplift, highest seed (pp)", max(r["up10"] for r in T), 2.3, 1, SRC)
check("RQ1", "T+Y: top-10% uplift, lowest seed (pp)", min(r["up10"] for r in TY), 2.7, 1, SRC)
check("RQ1", "T+Y: top-10% uplift, highest seed (pp)", max(r["up10"] for r in TY), 4.0, 1, SRC)
check("RQ4", "T: mean predicted effect, closest to zero (pp)", max(r["score"] for r in T), -5, 0, SRC)
check("RQ4", "T: mean predicted effect, farthest from zero (pp)", min(r["score"] for r in T), -10, 0, SRC)
sds = {m: np.std([holdout(m, "balanced_treatment", 4096, s)["q"] for s in SEEDS], ddof=1) for m in ("causalpfn", "tabpfn_t")}
check("Discussion", "context SD of Q-hat, smaller of CausalPFN / TabPFN-T (T, K=4,096)", min(sds.values()), 0.04, 2, SRC)
check("Discussion", "context SD of Q-hat, larger of CausalPFN / TabPFN-T (T, K=4,096)", max(sds.values()), 0.05, 2, SRC)

extra = pd.read_csv(CMP / "paper_extra_numbers.csv").set_index("quantity")["value"]
NPZ = "(from npz, via rerun_compare/paper_extra_numbers.csv)"
check("Setup", "holdout average effect (pp)", extra["full holdout ATE (pp)"], 1.01, 2, NPZ)
check("Introduction", "holdout visit rate (%)", 100 * extra["full holdout visit rate"], 4.7, 1, NPZ)
for name, lo, hi in (("T", 2.3, 2.9), ("TY", 0.3, 0.7)):
    bottom = [extra[f"full holdout {name} seed {s}: uplift bottom 10% (pp)"] for s in SEEDS]
    check("RQ1", f"{name}: bottom-10% uplift, lowest seed (pp)", min(bottom), lo, 1, NPZ)
    check("RQ1", f"{name}: bottom-10% uplift, highest seed (pp)", max(bottom), hi, 1, NPZ)
check("RQ4", "20%: mean predicted effect over 10 contexts (pp)",
      extra["subsample 20%: mean prediction over 10 contexts (pp)"], 6.8, 1, NPZ)

# ---- Fixed 200k-row holdout subsample (Table 1, RQ3, RQ4) ----------------------------------------------------
hs = pd.read_csv(CMP / "holdout_share_runs.csv")
h10 = pd.read_csv(CMP / "hs10_runs.csv")
S3, S10 = "rerun_compare/holdout_share_runs.csv", "rerun_compare/hs10_runs.csv"


def arm(df, column, name):
    return df[df[column] == name].sort_values("seed").set_index("seed")


for name, df, column, key, src, printed in (("T", hs, "strategy", "T", S3, (-0.053, -0.021, 1.43)),
                                            ("35%", hs, "strategy", "35%", S3, (-0.007, -0.003, 2.12)),
                                            ("T+Y", h10, "arm", "TY", S10, (0.049, 0.019, 3.50)),
                                            ("20%", h10, "arm", "20%", S10, (0.074, 0.029, 4.67)),
                                            ("HistGB S-learner, 2M rows", hs, "strategy", "ref_tau_s", S3, (0.082, 0.032, 5.36))):
    a = arm(df, column, key)
    for metric, value, dec in zip(("qhat", "auuc", "up10"), printed, (3, 3, 2)):
        check("Table 1, subsample", f"{name} (n={len(a)}): {metric}", a[metric].mean(), value, dec, src)
p20, pty = arm(h10, "arm", "20%"), arm(h10, "arm", "TY")
for metric, printed, dec in (("qhat", (0.025, 0.015, 0.034), 3), ("auuc", (0.010, 0.006, 0.014), 3),
                             ("up10", (1.17, 0.32, 2.02), 2)):
    check_ci("Table 1, subsample", f"20% - T+Y: {metric}, paired by seed", p20[metric] - pty[metric], printed, dec, S10)
check("RQ4", "contexts where 20% beats T+Y in Q-hat (of 10)", int((p20.qhat > pty.qhat).sum()), 10, 0, S10)
check_ci("RQ4", "20% - T+Y: uplift@10% (pp)", p20.up10 - pty.up10, (1.2, 0.3, 2.0), 1, S10)
ref_s = arm(hs, "strategy", "ref_tau_s").qhat.iloc[0]
check_ci("RQ4", "HistGB S-learner minus 20%: Q-hat, over contexts", ref_s - p20.qhat, (0.008, 0.003, 0.013), 3, S10)
first20, first35, firstty = arm(hs, "strategy", "20%"), arm(hs, "strategy", "35%"), arm(hs, "strategy", "TY")
check_ci("RQ3", "first-check 20% minus 35%: Q-hat, paired by seed", first20.qhat - first35.qhat, (0.082, 0.058, 0.107), 3, S3)
check_ci("Setup, holdout use", "first check 20% - T+Y (3 contexts): Q-hat", first20.qhat - firstty.qhat,
         (0.024, -0.003, 0.052), 3, S3)
for name in ("T", "35%"):
    rho = arm(hs, "strategy", name).rho_mu0
    check("RQ3", f"holdout {name}: contexts with rho < 0 (of {len(rho)})", int((rho < 0).sum()), len(rho), 0, S3)
    check("RQ3", f"holdout {name}: mean rho", rho.mean(), None, 2, S3)
check("RQ3", "holdout T and 35% together: mean rho",
      pd.concat([arm(hs, "strategy", n).rho_mu0 for n in ("T", "35%")]).mean(), -0.47, 2, S3)
check("RQ3", "holdout 20%: contexts with rho > 0 (of 10)", int((p20.rho_mu0 > 0).sum()), 10, 0, S10)
check("RQ3", "holdout 20%: mean rho", p20.rho_mu0.mean(), 0.39, 2, S10)

boot = pd.read_csv(CMP / "hs10_contrasts.csv").set_index("contrast")
B = "(from npz, via rerun_compare/hs10_contrasts.csv: two-level bootstrap)"
check("Abstract, RQ4", "20% - T+Y: Q-hat, two-level low", boot.loc["20% - TY, qhat", "low"], 0.005, 3, B)
check("Abstract, RQ4", "20% - T+Y: Q-hat, two-level high", boot.loc["20% - TY, qhat", "high"], 0.043, 3, B)
check("RQ4", "20% - T+Y: uplift@10%, two-level low (pp)", boot.loc["20% - TY, up10 pp", "low"], -0.1, 1, B)
check("RQ4", "20% - T+Y: uplift@10%, two-level high (pp)", boot.loc["20% - TY, up10 pp", "high"], 2.4, 1, B)
check("RQ4", "HistGB S-learner minus 20%: two-level low", -boot.loc["20% - ref S-learner, qhat", "high"], -0.006, 3, B)
check("RQ4", "HistGB S-learner minus 20%: two-level high", -boot.loc["20% - ref S-learner, qhat", "low"], 0.022, 3, B)

share20_runs = sorted(DSRUNS.glob("hs_*/unpacked/repro/home18/projects/smiles/results/fm_holdout_share/criteo/holdout_dopfn_share20_*.json"))
if share20_runs:
    scores = [100 * json.loads(p.read_text())["score_mean"] for p in share20_runs]
    check("RQ4", f"20%: mean predicted effect from the {len(scores)} DataSphere run JSONs (pp)", np.mean(scores), 6.8, 1,
          "ds/results/hs_*/unpacked/.../holdout_dopfn_share20_*.json")

# ---- Criteo and LZD cross-validation (abstract, RQ1-RQ3, Fig. 1-2) -------------------------------------------
cv = []
for p in CV18.glob("fold*_dopfn_*_eval8000_cuda0.json"):
    d = json.loads(p.read_text())
    cv.append(dict(strategy=STRAT[d["strategy"]], k=d["context_size"], fold=int(p.name[4]), q=d["metrics"]["qini_auc_normalized"]))
cv = pd.DataFrame(cv)
by_k = cv.groupby(["strategy", "k"]).q.mean()
C18 = "home18/projects/smiles/results/fm_cv/criteo/fold*_dopfn_*.json"
check("RQ1, Fig. 2a", "R and T, K 512-4,096: lowest mean Q-hat", by_k[["R", "T"]].min(), -0.085, 3, C18)
check("RQ1, Fig. 2a", "R and T, K 512-4,096: highest mean Q-hat", by_k[["R", "T"]].max(), -0.072, 3, C18)
check("RQ1, Fig. 2a", "T+Y, K 512-4,096: lowest mean Q-hat", by_k["TY"].min(), 0.037, 3, C18)
check("RQ1, Fig. 2a", "T+Y, K 512-4,096: highest mean Q-hat", by_k["TY"].max(), 0.056, 3, C18)

mr = pd.read_csv(CMP / "mechanism_runs.csv")
M = "rerun_compare/mechanism_runs.csv"


def runs(dataset, model, strategy):
    return mr[(mr.dataset == dataset) & (mr.model == model) & (mr.strategy == strategy)]


def folds(dataset, model, strategy, column="qhat"):
    return runs(dataset, model, strategy).groupby("fold")[column].mean().sort_index().to_numpy()


check("Abstract, RQ1", "Criteo T, K=1,024: Q-hat", folds("criteo", "dopfn", "T").mean(), -0.077, 3, M)
check("Abstract", "Criteo T+Y, K=1,024: Q-hat", folds("criteo", "dopfn", "TY").mean(), 0.056, 3, M)
check("RQ1", "Criteo T: folds with negative mean (of 5)", int((folds("criteo", "dopfn", "T") < 0).sum()), 5, 0, M)
check("RQ1", "Criteo T: runs with negative Q-hat (of 15)", int((runs("criteo", "dopfn", "T").qhat < 0).sum()), 14, 0, M)
gain = {m: folds("criteo", m, "TY") - folds("criteo", m, "T") for m in ("dopfn", "causalpfn", "tabpfn_t")}
check_ci("RQ1, Fig. 2b", "Criteo Do-PFN T+Y - T, fold level", gain["dopfn"], (0.133, 0.102, 0.164), 3, M)
check("RQ2, Fig. 2b", "CausalPFN T+Y - T", gain["causalpfn"].mean(), 0.006, 3, M)
check("RQ2, Fig. 2b", "TabPFN-T T+Y - T", gain["tabpfn_t"].mean(), -0.010, 3, M)
check_ci("RQ2", "Do-PFN gain minus CausalPFN gain, fold level", gain["dopfn"] - gain["causalpfn"], (0.127, 0.044, 0.210), 3, M)
check_ci("RQ2", "Do-PFN gain minus TabPFN-T gain, fold level", gain["dopfn"] - gain["tabpfn_t"], (0.143, 0.062, 0.223), 3, M)
learner = {(c, s): folds("criteo", "dopfn", s, c) for c in ("ctx_logit_T_qhat", "ctx_hgb_S_qhat") for s in ("R", "T", "TY")}
check("RQ2", "same-row learners: lowest mean Q-hat (R, T, T+Y)", min(v.mean() for v in learner.values()), 0.005, 3, M)
check("RQ2", "same-row learners: highest mean Q-hat (R, T, T+Y)", max(v.mean() for v in learner.values()), 0.025, 3, M)
check("RQ2", "same-row learners: intervals that include zero (of 6)",
      sum(tci(v)[1] <= 0 <= tci(v)[2] for v in learner.values()), 6, 0, M)
negative_folds = {f"{m} {s}": int((folds("criteo", m, s) < 0).sum()) for m in ("causalpfn", "tabpfn_t") for s in ("R", "T")}
negative_folds.update({f"{c} {s}": int((learner[(c, s)] < 0).sum()) for c in ("ctx_logit_T_qhat", "ctx_hgb_S_qhat") for s in ("R", "T")})
check("Fig. 2 caption", "Do-PFN R and T: folds with negative mean (of 10)",
      int((folds("criteo", "dopfn", "R") < 0).sum() + (folds("criteo", "dopfn", "T") < 0).sum()), 10, 0, M)
check("Fig. 2 caption", "other models, R or T: most folds with negative mean", max(negative_folds.values()), 2, 0, M)
for share, printed in (("10%", -0.051), ("20%", 0.087), ("35%", 0.021), ("TY", 0.056), ("75%", 0.066)):
    check("RQ3, Fig. 1a", f"Criteo CV Q-hat at share {share}", folds("criteo", "dopfn", share).mean(), printed, 3, M)
check_ci("RQ3", "Criteo CV Do-PFN 20% - T+Y", folds("criteo", "dopfn", "20%") - folds("criteo", "dopfn", "TY"),
         (0.031, 0.006, 0.057), 3, M)
logit = folds("criteo", "dopfn", "20%", "ctx_logit_T_qhat") - folds("criteo", "dopfn", "TY", "ctx_logit_T_qhat")
mean, low, high = tci(logit)
check("RQ3", "Criteo CV logistic T-learner 20% - T+Y", mean, 0.035, 3, M)
check("RQ3", "Criteo CV logistic T-learner 20% - T+Y: interval half-width", (high - low) / 2, 0.11, 2, M)
check("RQ3", "Criteo CV HistGB S-learner 20% - T+Y",
      (folds("criteo", "dopfn", "20%", "ctx_hgb_S_qhat") - folds("criteo", "dopfn", "TY", "ctx_hgb_S_qhat")).mean(), 0.003, 3, M)
for share in ("R", "T", "10%", "35%"):
    rho = runs("criteo", "dopfn", share).rho_mu0
    check("RQ3", f"Criteo CV {share}: runs with rho < 0 (of {len(rho)})", int((rho < 0).sum()), len(rho), 0, M)
check("RQ3", "Criteo CV R: mean rho", runs("criteo", "dopfn", "R").rho_mu0.mean(), -0.70, 2, M)
rho20 = runs("criteo", "dopfn", "20%").rho_mu0
check("RQ3", f"Criteo CV 20%: runs with rho > 0 (of {len(rho20)})", int((rho20 > 0).sum()), len(rho20), 0, M)
check("RQ3", "Criteo CV 20%: mean rho", rho20.mean(), 0.47, 2, M)
check("RQ3", "Criteo CV 75%: mean rho", runs("criteo", "dopfn", "75%").rho_mu0.mean(), 0.13, 2, M)
check("RQ3", "Criteo CV T+Y: mean rho ('about 0')", runs("criteo", "dopfn", "TY").rho_mu0.mean(), None, 2, M)
check("RQ1", "LZD T: Q-hat", folds("lzd", "dopfn", "T").mean(), -0.016, 3, M)
check("RQ1", "LZD T+Y: Q-hat", folds("lzd", "dopfn", "TY").mean(), 0.016, 3, M)
check_ci("RQ1", "LZD T+Y - T, fold level", folds("lzd", "dopfn", "TY") - folds("lzd", "dopfn", "T"), (0.032, -0.027, 0.090), 3, M)
check_ci("RQ4", "LZD 20% - T+Y, fold level", folds("lzd", "dopfn", "20%") - folds("lzd", "dopfn", "TY"), (-0.026, -0.075, 0.022), 3, M)
check("RQ3", "LZD 10%: mean rho", runs("lzd", "dopfn", "10%").rho_mu0.mean(), 0.19, 2, M)
check("RQ3", "LZD 20%: mean rho", runs("lzd", "dopfn", "20%").rho_mu0.mean(), 0.30, 2, M)
check("Setup", "LZD T, K=1,024 (Section 2 quotes -0.016)", folds("lzd", "dopfn", "T").mean(), -0.016, 3, M)

# ---- Fig. 3 (plotted points; the per-context arrays are in rerun_compare/*.npz) -------------------------------
f3 = pd.read_csv(ROOT / "paper/figures/fig3_points.csv").set_index("decile")
F3 = "paper/figures/fig3_points.csv"
check("RQ3, Fig. 3a", "uplift in decile 9 (pp)", f3.loc[9, "uplift_pp"], 2.4, 1, F3)
check("RQ3, Fig. 3a", "uplift in decile 10 (pp)", f3.loc[10, "uplift_pp"], 4.3, 1, F3)
for column, printed in (("pct_T", 24), ("pct_TY", 63), ("pct_20%", 87)):
    check("RQ3, Fig. 3b", f"top decile, mean score percentile ({column[4:]})", f3.loc[10, column], printed, 0, F3)
check("RQ3, Fig. 3b", "T+Y deciles 1, 9, 10: lowest percentile", f3.loc[[1, 9, 10], "pct_TY"].min(), 61, 0, F3)
check("RQ3, Fig. 3b", "T+Y deciles 1, 9, 10: highest percentile", f3.loc[[1, 9, 10], "pct_TY"].max(), 63, 0, F3)
check("RQ3, Fig. 3b", "T+Y deciles 6-7: lowest percentile", f3.loc[[6, 7], "pct_TY"].min(), 31, 0, F3)
check("RQ3, Fig. 3b", "T+Y deciles 6-7: highest percentile", f3.loc[[6, 7], "pct_TY"].max(), 33, 0, F3)

out = pd.DataFrame(rows)
out.to_csv(CMP / "paper_numbers.csv", index=False)
bad = out[out.ok == False]  # noqa: E712 (None means "reported only")
for r in out.itertuples():
    flag = "  " if r.ok is None else ("ok" if r.ok else "!!")
    printed = "" if r.paper is None else f"  paper {r.paper:g}"
    print(f"{flag} {r.section:<24} {r.quantity:<70} {r.value:10.4f}{printed}")
print(f"\n{len(out)} numbers, {int(out.ok.eq(True).sum())} match the paper, {len(bad)} differ, "
      f"{int(out.ok.isna().sum())} reported only -> {CMP / 'paper_numbers.csv'}")
sys.exit(1 if len(bad) else 0)
