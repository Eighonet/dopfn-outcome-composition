"""Headline numbers of the re-check, stated at fold level. Reads the CSVs written by the other scripts; nothing is recomputed.

Five evaluation sets per dataset and fold split, so every count is "k of 5 folds" (fold mean over context seeds and sizes)
and every interval is a t-interval over the five fold means (4 df), as in the paper.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student

HERE = Path(__file__).resolve().parent
ORDER = ["R", "T", "10%", "20%", "35%", "TY", "75%"]
MODELS = ["dopfn", "causalpfn", "tabpfn_t"]
SETS = ["criteo", "lzd", "hillstrom", "retailhero", "orange"]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
keys = ["ds", "model", "strategy", "K", "fold", "seed"]
cols = ["q", "q_neg", "risk_q", "mu0_q", "q_between", "q_within", "rho"]

# one table of Q, negated Q, the same-rows risk ranking and the decomposition for every CV run
home = pd.concat([pd.read_csv(HERE / "recheck_five_small.csv"), pd.read_csv(HERE / "recheck_five_grouped.csv")]).assign(tree="home")
h18 = pd.read_csv(HERE / "recheck_five_h18.csv").assign(tree="home18")
cv = pd.concat([h18, home], ignore_index=True)[["tree"] + keys + cols]
assert cv[cols].notna().all().all() and len(cv) == 3900

# the earlier per-dataset tables of the paper's folds (K=1024 only for the risk ranking) must agree with the uniform one
old = pd.concat([pd.read_csv(HERE / "recheck_cv.csv").assign(ds="criteo"), pd.read_csv(HERE / "recheck_lzd.csv").assign(ds="lzd")])
old = old.rename(columns={"qhat": "q", "risk_logit_q": "risk_q"})[keys + ["q", "risk_q", "mu0_q", "q_between", "q_within"]]
m = h18.merge(old, on=keys, suffixes=("", "_old"), validate="one_to_one")
n = h18.merge(pd.read_csv(HERE / "recheck_negated.csv")[keys + ["q_neg"]], on=keys, suffixes=("", "_old"), validate="one_to_one")
diff = {c: float((m[c] - m[c + "_old"]).abs().max()) for c in ["q", "risk_q", "mu0_q", "q_between", "q_within"]}
diff["q_neg"] = float((n.q_neg - n.q_neg_old).abs().max())
print("cross-check with the earlier tables, max abs difference:", {k: f"{v:.1e}" for k, v in diff.items()})
assert len(m) == len(n) == 1200 and max(diff["q"], diff["risk_q"], diff["mu0_q"], diff["q_neg"]) < 1e-9
# mu0-hat is identical, so the between/within numbers differ only through the random order of tied mu0-hat values at the stratum borders
print("  between/within, same mu0-hat, different random tie order: mean abs difference by dataset")
print(m.assign(between=(m.q_between - m.q_between_old).abs(), within=(m.q_within - m.q_within_old).abs())
      .groupby("ds")[["between", "within"]].agg(["mean", "max"]).round(4).to_string())


def tint(v):
    """Mean and 95% t-interval over the fold values."""
    v = np.asarray(v, dtype=float)
    h = student.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():+.3f} [{v.mean() - h:+.3f}, {v.mean() + h:+.3f}]"


def folds(d, by):
    """Fold means, then counts over the five folds."""
    f = d.groupby(by + ["fold"])[["q", "q_neg", "risk_q", "q_between", "q_within", "rho"]].mean()
    g = f.groupby(by)
    return pd.DataFrame(dict(
        q=g.q.mean(), q_min=g.q.min(), q_max=g.q.max(), folds_q_neg=g.q.agg(lambda v: int((v < 0).sum())),
        negated=g.q_neg.mean(), risk=g.risk_q.mean(), folds_below_risk=f.assign(b=f.q < f.risk_q).groupby(by).b.sum(),
        folds_negated_below_risk=f.assign(b=f.q_neg < f.risk_q).groupby(by).b.sum(),
        between=g.q_between.mean(), within=g.q_within.mean(), rho=g.rho.mean())), f


for title, d in [
    ("Criteo, the paper's folds, K=1024 (all seven designs)", cv[(cv.tree == "home18") & (cv.ds == "criteo") & (cv.K == 1024)]),
    ("Criteo, the paper's folds, all K (R, T, T+Y)", cv[(cv.tree == "home18") & (cv.ds == "criteo") & cv.strategy.isin(["R", "T", "TY"])]),
    ("Criteo, second fold split, all K", cv[(cv.tree == "home") & (cv.ds == "criteo")]),
    ("LZD, the paper's folds, K=1024 (all seven designs)", cv[(cv.tree == "home18") & (cv.ds == "lzd") & (cv.K == 1024)]),
    ("LZD, the paper's folds, all K (R, T, T+Y)", cv[(cv.tree == "home18") & (cv.ds == "lzd") & cv.strategy.isin(["R", "T", "TY"])]),
    ("LZD, second fold split, all K", cv[(cv.tree == "home") & (cv.ds == "lzd")]),
    ("Hillstrom, all K", cv[cv.ds == "hillstrom"]),
    ("RetailHero, all K", cv[cv.ds == "retailhero"]),
    ("Orange, all K", cv[cv.ds == "orange"]),
]:
    t, f = folds(d, ["model", "strategy"])
    idx = [(a, s) for a in MODELS for s in ORDER if (a, s) in t.index]
    print(f"\n##### {title}; Q(mu0) = {d.groupby('fold').mu0_q.first().mean():.3f}, SD over the five eval sets {d.groupby('fold').mu0_q.first().std():.3f}")
    print(t.reindex(idx).round(3).to_string())
    print("  fold-level difference to the same-rows risk ranking, mean [95% t-interval]:")
    for a, s in idx:
        e = f.loc[(a, s)]
        print(f"    {a:10s} {s:4s} {tint(e.q - e.risk_q)}" + (f"   negated: {tint(e.q_neg - e.risk_q)}" if a == "dopfn" else ""))
    ty = f.loc[("dopfn", "TY")]
    for s in ["R", "T"]:
        e = f.loc[("dopfn", s)]
        print(f"  Do-PFN: negated {s} score against T+Y, by fold: {np.round((e.q_neg - ty.q).to_numpy(), 3)} -> negated at least as good in "
              f"{int((e.q_neg >= ty.q).sum())} of 5; {tint(e.q_neg - ty.q)}")

print("\n##### Do-PFN against the same-rows risk ranking by context size (fold means over three context seeds)")
d = cv[(cv.model == "dopfn") & cv.ds.isin(["criteo", "lzd"]) & cv.strategy.isin(["R", "T", "TY"])]
f = d.groupby(["ds", "tree", "strategy", "K", "fold"])[["q", "q_neg", "risk_q"]].mean()
by = ["ds", "tree", "strategy", "K"]
g = f.groupby(by)
print(pd.DataFrame(dict(q=g.q.mean(), negated=g.q_neg.mean(), risk=g.risk_q.mean(), folds_q_neg=g.q.agg(lambda v: int((v < 0).sum())),
                        folds_below_risk=f.assign(b=f.q < f.risk_q).groupby(by).b.sum(),
                        folds_negated_below_risk=f.assign(b=f.q_neg < f.risk_q).groupby(by).b.sum())).round(3).to_string())

print("\n##### Criteo final holdout")
h = pd.read_csv(HERE / "recheck_holdout.csv")
print(h.groupby(["n_eval", "source", "model", "strategy", "K"]).agg(
    contexts=("q", "size"), q=("q", "mean"), q_sd=("q", "std"), negated=("q_neg", "mean"), risk=("risk_q", "mean"), mu0=("mu0_q", "mean"),
    below_risk=("q", lambda v: int((v < h.loc[v.index, "risk_q"]).sum())), rho=("rho", "mean"), ate_hat=("ate_hat", "mean"),
    z=("z_ate", "mean"), between=("q_between", "mean"), within=("q_within", "mean")).round(3).to_string())

# predicted average effect against the context's own difference in means, and the value range, for every CV run
ck = ["tree", "ds", "model", "strategy", "K", "fold", "seed"]
c = pd.read_csv(HERE / "recheck_ate_context.csv").merge(pd.read_csv(HERE / "recheck_range.csv")[ck + ["above1", "max_abs", "median"]], on=ck, validate="one_to_one")
assert len(c) == 3900
c["flag"] = c.z.abs() > 2
c["flag_sign"] = c.flag & (np.sign(c.ate_hat) != np.sign(c.ctx_dm))
c["flag_range"] = c.above1 > 0

print("\n##### Equal-share designs, K=1024, the paper's folds: the context carries no treatment-outcome contrast; Do-PFN's predicted average effect")
e = c[(c.tree == "home18") & (c.model == "dopfn") & (c.K == 1024) & ~c.strategy.isin(["R", "T"])]
print(e.groupby(["ds", "strategy"]).agg(contexts=("q", "size"), ctx_pos=("ctx_pos", "mean"), max_abs_contrast=("ctx_dm", lambda v: v.abs().max()),
                                        ate_hat=("ate_hat", "mean"), ate_min=("ate_hat", "min"), ate_max=("ate_hat", "max"),
                                        positive=("ate_hat", lambda v: int((v > 0).sum())), median_row=("median", "mean"), flagged=("flag", "sum"),
                                        rho=("rho", "mean"), q=("q", "mean"))
      .reindex([(a, s) for a in ["criteo", "lzd"] for s in ORDER[2:]]).round(3).to_string())

print("\n##### Context check on natural contexts (R, T), both fold splits: |predicted average effect - context difference in means| > 2 SE")
nat = c[c.strategy.isin(["R", "T"])]
f = nat.groupby(["ds", "model", "tree", "fold"]).agg(flag=("flag", "mean"), wrong=("wrong_sign", "mean"), q=("q", "mean")).reset_index()
t = nat.groupby(["ds", "model"]).agg(runs=("flag", "size"), flagged=("flag", "mean"), flagged_wrong_sign=("flag_sign", "mean"), wrong_sign=("wrong_sign", "mean"),
                                     out_of_range=("flag_range", "mean"), ate_hat=("ate_hat", "mean"), ctx_dm=("ctx_dm", "mean"), ratio=("ratio", "median"),
                                     q=("q", "mean"), runs_q_neg=("q", lambda v: (v < 0).mean()))
t["fold_cells"] = f.groupby(["ds", "model"]).size()
t["cells_mostly_flagged"] = f.assign(b=f.flag > 0.5).groupby(["ds", "model"]).b.sum()
t["cells_all_wrong_sign"] = f.assign(b=f.wrong == 1).groupby(["ds", "model"]).b.sum()
t["cells_q_neg"] = f.assign(b=f.q < 0).groupby(["ds", "model"]).b.sum()
print(t.reindex([(a, b) for a in SETS for b in MODELS]).round(3).to_string())

# equal-share designs are left out here: their context contrast is exactly zero, so a comparison with it says nothing
print("\n##### Natural contexts, all models: runs flagged by either check (|z| > 2 or a value outside [-1, 1]) against Q-hat")
nat = nat.assign(either=nat.flag | nat.flag_range)
print(nat.groupby(["ds", "model", "either"]).agg(runs=("q", "size"), q=("q", "mean"), runs_q_neg=("q", lambda v: (v < 0).mean()), rho=("rho", "mean"),
                                                 wrong_sign=("wrong_sign", "mean"))
      .reindex([(a, b, k) for a in SETS for b in MODELS for k in [False, True]]).dropna(how="all").round(3).to_string())
print("\n##### The same for the range check alone")
print(nat.groupby(["ds", "model", "flag_range"]).agg(runs=("q", "size"), q=("q", "mean"), runs_q_neg=("q", lambda v: (v < 0).mean()), rho=("rho", "mean"))
      .reindex([(a, b, k) for a in SETS for b in MODELS for k in [False, True]]).dropna(how="all").round(3).to_string())

print("\n##### Predicted effects outside [-1, 1] on a 0/1 outcome, natural contexts")
print(nat.groupby(["ds", "model"]).agg(runs=("above1", "size"), rows_above1=("above1", "mean"), runs_with_any=("flag_range", "mean"),
                                       max_abs=("max_abs", "max"), mean=("ate_hat", "mean"), median=("median", "mean"))
      .reindex([(a, b) for a in SETS for b in MODELS]).round(4).to_string())
