"""Predicted average effect against what the context itself says. Reads run records only (no data, no GPU).

For every saved CV run: arm sizes and positive rates of the context (`context_counts`), the context's difference
in means and its standard error, the model's mean predicted effect on the evaluation rows (`score_mean`), Q-hat.
rho(tau-hat, mu0-hat) is merged from the CSVs written by recheck_five.py, recheck_ate_sign.py and recheck_lzd.py.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
OUT = Path(sys.argv[1])
STRAT = {"random": "R", "balanced_treatment": "T", "balanced_treatment_outcome": "TY",
         "share10": "10%", "share20": "20%", "share35": "35%", "share75": "75%"}
# difference in means on the full data, as printed by recheck_five.py
ATE = {"criteo": 0.0103, "lzd": 0.0037, "hillstrom": 0.0609, "retailhero": 0.0332, "orange": 0.0028}

rows = []
for tree, sub in [("home", "fm_cv"), ("home18", "fm_cv"), ("home18", "fm_dose")]:
    for f in glob.glob(str(REPRO / tree / "projects/smiles/results" / sub / "*" / "fold*_eval8000_cuda0.json")):
        d = json.load(open(f))
        c = d["context_counts"]
        n1, n0 = c["T1Y0"] + c["T1Y1"], c["T0Y0"] + c["T0Y1"]
        p1, p0 = c["T1Y1"] / n1, c["T0Y1"] / n0
        s1, s0 = (c["T1Y1"] + 0.5) / (n1 + 1), (c["T0Y1"] + 0.5) / (n0 + 1)
        rows.append(dict(tree=tree, ds=d["dataset"], model=d["model"], strategy=STRAT[d["strategy"]], K=d["context_size"],
                         fold=d["fold"], seed=int(re.search(r"_seed(\d+)_", Path(f).name)[1]), n1=n1, n0=n0, p1=p1, p0=p0, ctx_pos=(c["T1Y1"] + c["T0Y1"]) / (n1 + n0),
                         ctx_dm=p1 - p0, ctx_se=np.sqrt(s1 * (1 - s1) / n1 + s0 * (1 - s0) / n0),
                         ate_hat=d["score_mean"], q=d["metrics"]["qini_auc_normalized"]))
df = pd.DataFrame(rows)

keys = ["tree", "ds", "model", "strategy", "K", "fold", "seed"]
rho = pd.concat([
    pd.concat([pd.read_csv(HERE / "recheck_five_small.csv"), pd.read_csv(HERE / "recheck_five_grouped.csv")]).assign(tree="home"),
    pd.read_csv(HERE / "recheck_ate_sign.csv").assign(tree="home18", ds="criteo"),
    pd.read_csv(HERE / "recheck_lzd.csv").assign(tree="home18", ds="lzd"),
])[keys + ["rho"]]
df = df.merge(rho, on=keys, how="left", validate="one_to_one")
assert df.rho.notna().all(), "runs without rho"
df["ate"] = df.ds.map(ATE)
df["z"] = (df.ate_hat - df.ctx_dm) / df.ctx_se
df["wrong_sign"] = np.sign(df.ate_hat) != np.sign(df.ate)
df["ratio"] = df.ate_hat.abs() / df.ate
df.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
order = ["R", "T", "10%", "20%", "35%", "TY", "75%"]

print("A. Do-PFN by dataset, tree and design (all K): the context's own difference in means, the predicted average effect, rho, Q-hat")
d = df[df.model == "dopfn"]
g = d.groupby(["ds", "tree", "strategy"]).agg(
    n=("q", "size"), ctx_pos=("ctx_pos", "mean"), ctx_dm=("ctx_dm", "mean"), max_abs_dm=("ctx_dm", lambda v: v.abs().max()),
    ate_hat=("ate_hat", "mean"), pos_ate=("ate_hat", lambda v: (v > 0).mean()), z=("z", "mean"), z2=("z", lambda v: (v.abs() > 2).mean()),
    rho=("rho", "mean"), pos_rho=("rho", lambda v: (v > 0).mean()), q=("q", "mean"), q_neg=("q", lambda v: (v < 0).mean()))
idx = [(a, b, s) for a in ["criteo", "lzd", "hillstrom", "retailhero", "orange"] for b in ["home18", "home"] for s in order if (a, b, s) in g.index]
print(g.reindex(idx).round(3).to_string())

print("\nB. Natural composition (R, T): does the predicted average effect follow the context's difference in means?")
print("   Spearman within (dataset, tree, design, K), averaged over those cells; slope of ate_hat on ctx_dm pooled after centring per cell")
nat = df[df.strategy.isin(["R", "T"])]
out = []
for (ds, model), e in nat.groupby(["ds", "model"]):
    cells = [c for _, c in e.groupby(["tree", "strategy", "K"])]
    sp = np.mean([spearmanr(c.ate_hat, c.ctx_dm).statistic for c in cells])
    cx = np.concatenate([c.ctx_dm - c.ctx_dm.mean() for c in cells])
    cy = np.concatenate([c.ate_hat - c.ate_hat.mean() for c in cells])
    out.append(dict(ds=ds, model=model, runs=len(e), spearman=sp, slope=np.polyfit(cx, cy, 1)[0],
                    wrong_sign=e.wrong_sign.mean(), median_ratio=e.ratio.median(), z2=(e.z.abs() > 2).mean(), q=e.q.mean(), q_neg=(e.q < 0).mean(),
                    abs_rho=e.rho.abs().mean()))
print(pd.DataFrame(out).set_index(["ds", "model"]).reindex(
    [(a, m) for a in ["criteo", "lzd", "hillstrom", "retailhero", "orange"] for m in ["dopfn", "causalpfn", "tabpfn_t"]]).round(3).to_string())

print("\nC. Do-PFN, natural composition, by K: predicted average effect and how often it has the wrong sign")
print(nat[nat.model == "dopfn"].groupby(["ds", "strategy", "K"]).agg(
    n=("q", "size"), ate_hat=("ate_hat", "mean"), wrong_sign=("wrong_sign", "mean"), z=("z", "mean"), rho=("rho", "mean"), q=("q", "mean"))
      .round(3).to_string())

print("\nD. Do-PFN, all runs: sign of Q-hat against (i) the sign of the predicted average effect, (ii) the sign rule")
d = d.assign(rule=np.sign(d.ate_hat) * np.sign(0.5 - d.ctx_pos))
for ds, e in d.groupby("ds"):
    e2 = e[e.strategy != "TY"]
    a = (np.sign(e2.q) == np.sign(e2.ate_hat) * np.sign(e2.ate)).mean()
    b = (np.sign(e2.q) == e2.rule * np.sign(e2.ate)).mean()
    c = (np.sign(e2.rho) == e2.rule).mean()
    print(f"  {ds:10s} runs {len(e2):4d} (T+Y excluded)  Q sign = predicted-effect sign: {a:.3f}   Q sign = rule: {b:.3f}   rho sign = rule: {c:.3f}")
