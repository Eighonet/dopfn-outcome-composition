"""Draft of the proposed main figure: Do-PFN against a risk-only ranking fitted on the same context rows (Criteo).

Reads recheck_five_h18.csv (the paper's folds) and recheck_five_grouped.csv (second fold split); no model is run.
Style and color encoding follow paper/figures/make_figures.py (color = context design). Not used by the paper yet.
  a  K=1,024, the paper's folds: Do-PFN's Q-hat per fold and on average, its negated score under R and T, and the
     Q-hat of a logistic regression of the outcome on the features (no treatment column) fitted on the same rows
  b  all K: Q-hat minus the risk-only Q-hat, mean and 95% t-interval over the five folds, on both fold splits
"""
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as tdist

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "recheck_fig_risk_reference"

INK, INK2, GRID, TINT = "#0b0b0b", "#52514e", "#e4e3df", "#f6e7df"
C_T, C_TY, C_20, C_GRAY, C_LIGHT = "#eb6834", "#2a78d6", "#1baf7a", "#8a8984", "#bdbcb6"
COLOR = {"R": C_GRAY, "T": C_T, "10%": C_GRAY, "20%": C_20, "35%": C_GRAY, "TY": C_TY, "75%": C_GRAY}
FULL = 7.0                           # acmart sigconf text width, inches

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


def ci(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


h18 = pd.read_csv(HERE / "recheck_five_h18.csv")
grp = pd.read_csv(HERE / "recheck_five_grouped.csv")

# ---------------------------------------------------------------- panel a: by design, K=1,024, the paper's folds
ORDER = ["R", "T", "10%", "20%", "35%", "TY", "75%"]
XPOS = dict(zip(ORDER, [0.0, 1.0, 2.5, 3.5, 4.5, 5.5, 6.5]))
a = h18[(h18.ds == "criteo") & (h18.model == "dopfn") & (h18.K == 1024)]
points = []
for s in ORDER:
    f = a[a.strategy == s].groupby("fold")[["q", "q_neg", "risk_q"]].mean()
    assert len(f) == 5 and len(a[a.strategy == s]) == 15
    (q, qh), (n, nh), (r, rh) = ci(f.q), ci(f.q_neg), ci(f.risk_q)
    points.append(dict(strategy=s, q=q, q_half=qh, folds=f.q.to_numpy(), neg=n, neg_half=nh, neg_folds=f.q_neg.to_numpy(),
                       risk=r, risk_half=rh, risk_folds=f.risk_q.to_numpy(), below=int((f.q < f.risk_q).sum()),
                       negative=int((f.q < 0).sum()), neg_below=int((f.q_neg < f.risk_q).sum())))
points = pd.DataFrame(points)

# ---------------------------------------------------------------- panel b: gap to the risk-only ranking, all K
ROWS = [("Do-PFN", "dopfn", "q"), ("Do-PFN, negated", "dopfn", "q_neg"), ("CausalPFN", "causalpfn", "q"),
        ("TabPFN-T", "tabpfn_t", "q")]
gaps = []
for split, src in (("paper", h18), ("second", grp)):
    d = src[(src.ds == "criteo") & src.strategy.isin(["R", "T", "TY"])]
    for label, model, col in ROWS:
        for s in ("R", "T", "TY"):
            if col == "q_neg" and s == "TY":
                continue                                             # negation is the remedy for the natural designs
            e = d[(d.model == model) & (d.strategy == s)]
            f = e.groupby("fold")[[col, "risk_q"]].mean()
            assert len(f) == 5 and len(e) == 60
            m, h = ci(f[col] - f.risk_q)
            gaps.append(dict(split=split, row=label, strategy=s, gap=m, half=h, lo=m - h, hi=m + h))
gaps = pd.DataFrame(gaps)

pd.set_option("display.width", 220)
print(points.drop(columns=["folds", "neg_folds", "risk_folds"]).round(3).to_string(index=False))
print(gaps.round(3).to_string(index=False))

# ---------------------------------------------------------------- figure
fig, (ax, bx) = plt.subplots(1, 2, figsize=(FULL, 2.62), layout="constrained", gridspec_kw=dict(width_ratios=[1.5, 1]))

ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
ax.axhspan(-0.15, 0, color=TINT, zorder=0, linewidth=0)
ax.axhline(0, color=INK2, linewidth=0.8, zorder=1)
COLS = np.linspace(-0.28, 0.28, 5)                # one column per fold: the three marks in a column share an eval set
for _, p in points.iterrows():
    x, c = XPOS[p.strategy], COLOR[p.strategy]
    for xc, q, r, n in zip(x + COLS, p.folds, p.risk_folds, p.neg_folds):
        ax.plot([xc, xc], [q, r], color=C_LIGHT, linewidth=0.7, zorder=2)
        ax.plot([xc - 0.075, xc + 0.075], [r, r], color=INK, linewidth=1.5, solid_capstyle="butt", zorder=4)
        if p.strategy in ("R", "T"):
            ax.plot(xc, n, "o", ms=3.5, mfc="white", mec=c, mew=0.9, zorder=5)
        ax.plot(xc, q, "o", ms=4.2, color=c, mec="white", mew=0.5, zorder=6)
ax.set_xlim(-0.6, 7.1)
ax.set_ylim(-0.15, 0.16)
ax.set_yticks([-0.1, -0.05, 0, 0.05, 0.1])
ax.set_xticks([XPOS[s] for s in ORDER])
ax.set_xticklabels(["R", "T", "10%", "20%", "35%", "T+Y\n(50%)", "75%"])
ax.set_ylabel("Normalized Qini")
ax.set_title("a  Do-PFN and a risk-only ranking, same context rows", pad=4)
ax.text(7.05, -0.143, "worse than random", fontsize=7, color=INK2, va="bottom", ha="right", style="italic")
tr = ax.get_xaxis_transform()
for lo, hi, text in ((-0.45, 1.45, "natural share of Y = 1"), (2.05, 6.95, "equal share of Y = 1 in both arms")):
    ax.plot([lo, hi], [-0.2, -0.2], color=INK2, linewidth=0.6, transform=tr, clip_on=False)
    ax.text((lo + hi) / 2, -0.235, text, transform=tr, ha="center", va="top", fontsize=7.5, color=INK)

ypos, labels, y = {}, [], 0.0
for label, _, col in ROWS:
    for s in ("R", "T", "TY"):
        if col == "q_neg" and s == "TY":
            continue
        ypos[(label, s)] = y
        name = {"R": "R", "T": "T", "TY": "T+Y"}[s]
        labels.append((y, f"{label}   {name}" if s == "R" else name))
        y += 1.0
    y += 0.55
    if label != ROWS[-1][0]:
        bx.axhline(y - 0.775, color=C_LIGHT, linewidth=0.5, zorder=0.5)
bx.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
bx.set_axisbelow(True)
bx.axvspan(-0.26, 0, color=TINT, zorder=0, linewidth=0)
bx.axvline(0, color=INK2, linewidth=0.8, zorder=1)
for _, g in gaps.iterrows():
    yy, c = ypos[(g.row, g.strategy)], COLOR[g.strategy]
    if g.split == "paper":
        bx.errorbar(g.gap, yy - 0.17, xerr=g.half, fmt="o", ms=4.4, color=c, mec="white", mew=0.6, elinewidth=0.9, capsize=1.6, zorder=5)
    else:
        bx.errorbar(g.gap, yy + 0.17, xerr=g.half, fmt="s", ms=3.6, mfc="white", mec=c, mew=0.9, ecolor=c, elinewidth=0.9, capsize=1.6, zorder=4)
bx.set_yticks([v for v, _ in labels])
bx.set_yticklabels([t for _, t in labels])
bx.tick_params(axis="y", length=0)
bx.set_ylim(y - 1.05, -0.6)
bx.set_xlim(-0.26, 0.07)
bx.set_xticks([-0.2, -0.1, 0])
bx.set_xlabel("Normalized Qini minus risk-only (all K)", labelpad=2)
bx.set_title("b  Gap to the risk-only ranking", pad=4)
bx.text(-0.255, y - 1.0, "below risk-only", fontsize=7, color=INK2, va="bottom", ha="left", style="italic")

handles = [mpl.lines.Line2D([], [], marker="o", ls="", color=C_GRAY, mec="white", ms=4.6, label="Do-PFN (a: one column per fold)"),
           mpl.lines.Line2D([], [], marker="o", ls="", mfc="white", mec=C_GRAY, mew=0.9, ms=3.8, label="Do-PFN, negated score"),
           mpl.lines.Line2D([], [], marker="_", ls="", color=INK, mew=1.5, ms=7, label="risk-only logistic regression, same rows"),
           mpl.lines.Line2D([], [], marker="s", ls="", mfc="white", mec=C_GRAY, mew=0.9, ms=3.6, label="b: second fold split")]
fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.2, fontsize=7.2)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
print("saved", OUT.with_suffix(".png"))
