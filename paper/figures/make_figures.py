"""Figures for the WSDM 2027 short paper (Do-PFN, outcome composition of the context, Criteo).

Reads only saved results under repro/ (no model is run) and writes vector PDFs plus 300-dpi PNG previews,
each laid out at its final printed size so LaTeX includes it at scale 1:
  fig1_share.pdf    Q-hat and rho(score, baseline risk) against the Y=1 share of a 1,024-row context
  fig2_models.pdf   (a) Do-PFN against context size K; (b) R / T / T+Y for three foundation models and two
                    learners fitted on the same rows (CV), with the T+Y - T gain of each
  fig3_anatomy.pdf  where the uplift is (by baseline-risk decile) and where Do-PFN ranks those rows
Encoding shared by all figures: circle = CV, diamond = holdout; color = context design
(T orange #eb6834, T+Y blue #2a78d6, 20% green #1baf7a, other designs gray; palette validated for CVD).
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "repro"
CMP = REPRO / "rerun_compare"
OUT = Path(__file__).resolve().parent

INK, INK2, GRID, TINT = "#0b0b0b", "#52514e", "#e4e3df", "#f6e7df"
C_T, C_TY, C_20, C_GRAY, C_LIGHT = "#eb6834", "#2a78d6", "#1baf7a", "#8a8984", "#bdbcb6"
SHARE_COLOR = {"T": C_T, "10%": C_GRAY, "20%": C_20, "35%": C_GRAY, "TY": C_TY, "75%": C_GRAY}
FULL, COLUMN = 7.0, 3.33             # acmart sigconf text and column widths, inches

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.dpi": 300, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
})
CV_MARK = dict(fmt="o", ms=4.4, mec="white", mew=0.6, elinewidth=0.9, capsize=1.6)
HO_MARK = dict(fmt="D", ms=4.0, mec=INK, mew=0.6, elinewidth=0.9, capsize=1.6)


def ci(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    half = tdist.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return v.mean(), half


def fold_ci(df, col):
    """Mean over seeds within each fold, then a t-interval over the 5 folds."""
    return ci(df.groupby("fold")[col].mean().to_numpy())


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def newcombe(k1, n1, k0, n0):
    """Difference of two proportions with Newcombe's hybrid score interval (stays sensible at zero counts)."""
    p1, l1, u1 = wilson(k1, n1)
    p0, l0, u0 = wilson(k0, n0)
    d = p1 - p0
    return d, d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- data
mech = pd.read_csv(CMP / "mechanism_runs.csv")
cv = mech[(mech.dataset == "criteo") & (mech.model == "dopfn")].copy()
cv["share"] = (cv.ctx_y_share_t0 + cv.ctx_y_share_t1) / 2
hs3 = pd.read_csv(CMP / "holdout_share_runs.csv")          # T (frozen, 3 seeds), 35% (3 seeds), references
hs10 = pd.read_csv(CMP / "hs10_runs.csv")                   # 20% and T+Y, 10 seeds, paired
boot = pd.read_csv(CMP / "hs10_contrasts.csv").set_index("contrast")   # two-level bootstrap (contexts x rows)
ref_s = float(hs3[hs3.strategy == "ref_tau_s"].qhat.iloc[0])
order = ["T", "10%", "20%", "35%", "TY", "75%"]
cv_pts = []
for s in order:
    d = cv[cv.strategy == s]
    q, qh = fold_ci(d, "qhat")
    r, rh = fold_ci(d, "rho_mu0")
    lq, lqh = fold_ci(d, "ctx_logit_T_qhat")
    cv_pts.append(dict(strategy=s, share=d.share.mean(), q=q, qh=qh, r=r, rh=rh, lq=lq, lqh=lqh))
cv_pts = pd.DataFrame(cv_pts)
hold = []
for s, src, arm in (("T", hs3, "T"), ("35%", hs3, "35%"), ("20%", hs10, "20%"), ("TY", hs10, "TY")):
    d = src[src.strategy == arm] if "strategy" in src else src[src.arm == arm]
    q, qh = ci(d.qhat)
    r, rh = ci(d.rho_mu0)
    hold.append(dict(strategy=s, share=float(cv_pts.set_index("strategy").share[s]), q=q, qh=qh, r=r, rh=rh, n=len(d)))
hold = pd.DataFrame(hold)
cv_pts.to_csv(OUT / "fig1_cv_points.csv", index=False)
hold.to_csv(OUT / "fig1_holdout_points.csv", index=False)

# ---------------------------------------------------------------- Figure 1
DODGE = 1.4                                   # CV circles left of the share, holdout diamonds right of it
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.32), layout="constrained")
for ax, key, ylab, title in ((axes[0], "q", "Normalized Qini", "a  Ranking quality"),
                             (axes[1], "r", r"Spearman $\rho$(score, $\hat\mu_0$)", "b  Alignment with baseline risk")):
    grid(ax)
    ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
    family = cv_pts[cv_pts.strategy != "T"]   # T is not an equal-share design, so it stays unconnected
    ax.plot(family.share * 100 - DODGE, family[key], color=C_LIGHT, linewidth=0.9, zorder=2)
    for _, p in cv_pts.iterrows():
        ax.errorbar(p.share * 100 - DODGE, p[key], yerr=p[key + "h"], color=SHARE_COLOR[p.strategy], zorder=4, **CV_MARK)
    for _, p in hold.iterrows():                # a white halo keeps the CV connector from running into the diamonds
        ax.plot(p.share * 100 + DODGE, p[key], "D", ms=6.6, color="white", zorder=4.6)
        ax.errorbar(p.share * 100 + DODGE, p[key], yerr=p[key + "h"], color=SHARE_COLOR[p.strategy], zorder=5, **HO_MARK)
    ax.set_xlim(0, 80)
    ax.set_xticks([cv_pts.share[0] * 100, 10, 20, 35, 50, 75])
    ax.set_xticklabels(["T\n(natural)", "10%", "20%", "35%", "T+Y\n(50%)", "75%"])
    ax.set_xlabel("Share of Y = 1 in each arm of the 1,024-row context", labelpad=2)
    ax.set_ylabel(ylab)
    ax.set_title(title, pad=4)
ax = axes[0]
fam = cv_pts[cv_pts.strategy != "T"]
ax.plot(fam.share * 100, fam.lq, color=C_GRAY, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
ax.plot(cv_pts.share[0] * 100, cv_pts.lq[0], marker="_", ms=7, mew=1.0, color=C_GRAY, zorder=3)
ax.axhline(ref_s, color=INK2, linewidth=0.9, linestyle=(0, (1, 1.6)), zorder=1)
ax.axhspan(-0.13, 0, color=TINT, zorder=0, linewidth=0)
ax.text(79, -0.124, "worse than random", fontsize=7, color=INK2, va="bottom", ha="right", style="italic")
ax.set_ylim(-0.13, 0.13)
hq = hold.set_index("strategy")
b = boot.loc["20% - TY, qhat"]
ax.annotate(f"20% − T+Y, holdout:\n+{b['mean']:.3f} [{b['low']:.3f}, {b['high']:.3f}]",
            xy=(20 + DODGE + 0.6, hq.q["20%"] + 0.004), xytext=(26.5, 0.113), fontsize=7, color=INK, ha="left",
            va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=1))
ax = axes[1]
ax.axhspan(-0.95, 0, color=TINT, zorder=0, linewidth=0)
ax.text(79, 0.76, "aligned", fontsize=7, color=INK2, va="top", ha="right", style="italic")
ax.text(79, -0.92, "anti-aligned", fontsize=7, color=INK2, va="bottom", ha="right", style="italic")
ax.set_ylim(-0.95, 0.8)
handles = [mpl.lines.Line2D([], [], marker="o", ls="", color=C_GRAY, mec="white", ms=4.4, label="Do-PFN, CV"),
           mpl.lines.Line2D([], [], marker="D", ls="", color=C_GRAY, mec=INK, ms=4.0, label="Do-PFN, holdout subsample"),
           mpl.lines.Line2D([], [], color=C_GRAY, lw=1.0, ls=(0, (3, 2)), label="(a) logistic T-learner, same rows"),
           mpl.lines.Line2D([], [], color=INK2, lw=0.9, ls=(0, (1, 1.6)), label="(a) HistGB S-learner, 2M rows")]
fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, handletextpad=0.4, columnspacing=1.6)
save(fig, "fig1_share")

# ---------------------------------------------------------------- Figure 2
rows = [("Do-PFN", "dopfn", "qhat"), ("CausalPFN", "causalpfn", "qhat"), ("TabPFN T-learner", "tabpfn_t", "qhat"),
        ("Logistic T-learner*", "dopfn", "ctx_logit_T_qhat"), ("HistGB S-learner*", "dopfn", "ctx_hgb_S_qhat")]
strat = [("R", "R (random)", C_GRAY), ("T", "T (treatment-balanced)", C_T), ("TY", "T+Y (outcome-balanced)", C_TY)]
fig = plt.figure(figsize=(FULL, 2.18), layout="constrained")
ak, ax, ad = fig.subplots(1, 3, gridspec_kw=dict(width_ratios=[1, 1.62, 0.16]))

# (a) Do-PFN against context size, from the report's CV runs (5 folds x 3 context seeds per K); home18 holds the
# sklearn-1.8 folds that match the report, as in novelty_b_baserate.py
kruns = []
for f in (REPRO / "home18/projects/smiles/results/fm_cv/criteo").glob("fold*_dopfn_*_eval8000_cuda0.json"):
    info = json.loads(f.read_text())
    kruns.append(dict(fold=info["fold"], strategy=info["strategy"], k=info["context_size"],
                      qhat=info["metrics"]["qini_auc_normalized"]))
kruns = pd.DataFrame(kruns)
assert kruns.groupby(["strategy", "k"]).size().eq(15).all() and len(kruns) == 180
grid(ak)
ak.axhspan(-0.14, 0, color=TINT, zorder=0, linewidth=0)
ak.axhline(0, color=INK2, linewidth=0.8, zorder=1)
ks = [512, 1024, 2048, 4096]
k_pts = []
for (s, _, color), name, dx in zip(strat, ("random", "balanced_treatment", "balanced_treatment_outcome"),
                                   (-0.07, 0.07, 0.0)):
    pts = [fold_ci(kruns[(kruns.strategy == name) & (kruns.k == k)], "qhat") for k in ks]
    xs_k = np.log2(ks) + dx
    ak.plot(xs_k, [p[0] for p in pts], color=color, linewidth=1.1, zorder=3)
    ak.errorbar(xs_k, [p[0] for p in pts], yerr=[p[1] for p in pts], color=color, zorder=4, **CV_MARK)
    k_pts += [dict(strategy=s, k=k, mean=m, half=h) for k, (m, h) in zip(ks, pts)]
ak.text(12.22, 0.056, "T+Y", color=C_TY, fontsize=7.5, va="center", ha="left", fontweight="bold")
ak.text(12.22, -0.072, "R", color=C_GRAY, fontsize=7.5, va="center", ha="left", fontweight="bold")
ak.text(12.22, -0.093, "T", color=C_T, fontsize=7.5, va="center", ha="left", fontweight="bold")
ak.text(10.5, -0.137, "worse than random", fontsize=7, color=INK2, va="bottom", ha="center", style="italic")
ak.set_xticks(np.log2(ks))
ak.set_xticklabels([f"{k:,}" for k in ks])
ak.set_xlim(8.75, 12.62)
ak.set_ylim(-0.14, 0.10)
ak.set_xlabel("Context size K (Do-PFN)", labelpad=2)
ak.set_ylabel("Normalized Qini")
ak.set_title("a  Context size does not fix the sign", pad=4)
pd.DataFrame(k_pts).to_csv(OUT / "fig2a_points.csv", index=False)

# (b) the same contexts read by other models and by learners fitted on those rows; (c) the T+Y - T gain of each
grid(ax, "x")
ax.axvspan(-0.13, 0, color=TINT, zorder=0, linewidth=0)
ax.axvline(0, color=INK2, linewidth=0.8, zorder=1)
fig2_rows = []
for i, (label, model, col) in enumerate(rows):
    y = len(rows) - 1 - i
    d = mech[(mech.dataset == "criteo") & (mech.model == model)]
    for j, (s, slabel, color) in enumerate(strat):
        m, h = fold_ci(d[d.strategy == s], col)
        ax.errorbar(m, y + (1 - j) * 0.22, xerr=h, color=color, zorder=4, label=slabel if i == 0 else None, **CV_MARK)
        fig2_rows.append(dict(row=label, strategy=s, mean=m, half=h))
    dd = d.pivot_table(index="fold", columns="strategy", values=col)
    g, gh = ci((dd["TY"] - dd["T"]).to_numpy())
    fig2_rows.append(dict(row=label, strategy="TY-T", mean=g, half=gh))
    ad.text(1.0, y, f"{g:+.3f}".replace("-", "−"), ha="right", va="center", fontsize=7.5, color=INK,
            fontweight="bold" if i == 0 else "normal")
ax.axhline(1.5, color=GRID, linewidth=0.8)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.13, 0.115)
ax.set_xticks([-0.10, -0.05, 0, 0.05, 0.10])
ax.set_ylim(-0.6, len(rows) - 0.3)
ax.set_xlabel("Normalized Qini (K = 1,024)", labelpad=2)
ax.set_title("b  Only Do-PFN is negative in every fold", pad=4)
ad.set_ylim(ax.get_ylim())
ad.set_xlim(0, 1)
ad.axis("off")
ad.set_title("Δ T+Y−T", loc="right", pad=4, fontsize=7.5, fontweight="normal", color=INK2)
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", frameon=False, ncol=3,
           handletextpad=0.3, columnspacing=1.6)
pd.DataFrame(fig2_rows).to_csv(OUT / "fig2_points.csv", index=False)
save(fig, "fig2_models")

# ---------------------------------------------------------------- Figure 3
cache = np.load(CMP / "holdout_share_cache.npz")
z10 = np.load(CMP / "hs10_scores.npz")
y, t, mu0 = cache["y"], cache["t"], cache["mu0"]
dec = pd.qcut(pd.Series(mu0).rank(method="first"), 10, labels=False).to_numpy()   # mu0 ties split by row order
up = []
for k in range(10):
    m = dec == k
    up.append(100 * np.array(newcombe(y[m & (t == 1)].sum(), (m & (t == 1)).sum(),
                                      y[m & (t == 0)].sum(), (m & (t == 0)).sum())))
up = np.array(up)
pct = lambda s: pd.Series(s).rank(pct=True).to_numpy() * 100
series = {"T": [cache[f"score_T_{s}"] for s in (42, 43, 44)],
          "TY": [z10[f"TY_{s}"] for s in range(42, 52)],
          "20%": [z10[f"20%_{s}"] for s in range(42, 52)]}
prof = {name: np.array([[pct(s)[dec == k].mean() for k in range(10)] for s in scores]) for name, scores in series.items()}
fig, (a1, a2) = plt.subplots(2, 1, figsize=(COLUMN, 2.72), sharex=True, layout="constrained",
                             gridspec_kw=dict(height_ratios=[1, 1.35]))
grid(a1); grid(a2)
xs = np.arange(1, 11)
a1.bar(xs, up[:, 0], width=0.72, color=[C_LIGHT] * 8 + [C_GRAY] * 2, edgecolor="none", zorder=3)
a1.errorbar(xs, up[:, 0], yerr=[up[:, 0] - up[:, 1], up[:, 2] - up[:, 0]], fmt="none", ecolor=INK2, elinewidth=0.7,
            capsize=1.4, zorder=4)
a1.axhline(0, color=INK2, linewidth=0.6, zorder=5)
a1.text(12.55, 4.3, "top two\ndeciles", fontsize=7, color=INK2, va="center", ha="right")
a1.set_ylabel("Uplift (pp)")
a1.set_title("a  Where the uplift is: treated minus control", pad=3)
a2.axhline(50, color=INK2, linewidth=0.6, linestyle=(0, (1, 1.6)))
a2.text(10.35, 47, "random", fontsize=7, color=INK2, va="top", ha="left", style="italic")
for name, color, label, dy in (("T", C_T, "T", 0), ("TY", C_TY, "T+Y", -3), ("20%", C_20, "20%", 2)):
    for row in prof[name]:
        a2.plot(xs, row, color=color, linewidth=0.5, alpha=0.35, zorder=2)
    mean = prof[name].mean(0)
    a2.plot(xs, mean, color=color, linewidth=1.4, marker="D", ms=3.0, mec=INK, mew=0.4, zorder=3)
    a2.text(10.35, mean[-1] + dy, label, color=INK, fontsize=7.5, va="center", ha="left")
a2.set_ylabel("Mean score percentile\n(100 = targeted first)")
a2.set_title("b  Where Do-PFN ranks those rows", pad=3)
a2.set_xticks(xs)
a2.set_xlabel(r"Decile of estimated baseline risk $\hat\mu_0$", labelpad=2)
a2.set_xlim(0.4, 12.6)
a2.set_ylim(0, 100)
a2.set_yticks([0, 25, 50, 75, 100])
save(fig, "fig3_anatomy")
pd.DataFrame(dict(decile=xs, uplift_pp=up[:, 0], low_pp=up[:, 1], high_pp=up[:, 2],
                  **{f"pct_{k}": v.mean(0) for k, v in prof.items()})).to_csv(OUT / "fig3_points.csv", index=False)
print("figures written to", OUT)
