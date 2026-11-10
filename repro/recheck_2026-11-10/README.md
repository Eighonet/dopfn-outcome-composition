# Re-check of the paper's analysis, 2026-11-10

CPU-only checks on saved predictions and run records, plus three small CPU pilots of Do-PFN on single contexts.
No GPU or DataSphere job was run, the paper was not edited, and nothing here is used by the paper yet.

- Verified fold-level numbers: `recheck_summary.txt` (written by `recheck_summary.py` from the CSVs).
- Assessment and improvement plan (Russian): `PLAN_ru.md`.
- Draft of the proposed main figure: `recheck_fig_risk_reference.png` / `.pdf` (`recheck_figure.py`).

## Conventions

- Two result trees. `home18/` holds Criteo and LZD on the paper's folds (scikit-learn 1.8.0), including the share
  designs of `fm_dose` (Do-PFN, K=1024). `home/` holds all five datasets on scikit-learn 1.5.2 folds; for Criteo and
  LZD that is a second, different fold split. The final Criteo holdout is the same in both trees.
- 5 folds x 3 context seeds give 15 contexts per design and K, but there is one 8,000-row evaluation sample per fold,
  shared by all seeds, designs and models: five distinct evaluation sets. Every statement is therefore made on fold
  means ("k of 5 folds"), with 95% t-intervals over the five fold values, as in the paper.
- The context seed is read from the file name (`_seed42_`). The JSON field `context_seed` is seed + fold.
- mu0-hat: HistGB (lr 0.05, 200 iterations) on control rows that are never evaluated: the final holdout for Criteo
  and LZD, all non-evaluated rows for the other datasets, the development part for the holdout check.
- "Risk-only ranking, same rows": logistic regression of the outcome on the features of the context rows, without the
  treatment column (median fill, standardised, C=1), scored on the same evaluation rows as the model.
- Negated score: -tau-hat. Predicted average effect: mean of tau-hat over the evaluation rows (`score_mean`).
- Between/within parts: 10 strata of mu0-hat. "Between" is the Q-hat of the score replaced by its stratum mean (only
  the order of the strata is kept); "within" is the Q-hat of the score replaced by its percentile rank inside the
  stratum (only the order inside the strata is kept). Two separate Qini values: they do not add up to Q-hat. Ties in
  mu0-hat are broken by a random jitter, so the two parts move with the call order (Criteo between: mean abs 0.0008,
  max 0.0053; LZD: 0.0040, max 0.0195). The paper needs a deterministic tie rule.

## Files

| File | Content |
|---|---|
| `recheck_five.py` -> `recheck_five_h18.csv`, `recheck_five_grouped.csv`, `recheck_five_small.csv` | One row per saved CV run: Q-hat, negated Q-hat, rho(tau-hat, mu0-hat), predicted average effect, context and eval difference in means, between/within parts, same-rows risk ranking. `h18` = Criteo and LZD on the paper's folds with the share designs (1,200 runs); `grouped` = Criteo and LZD on the second split (1,080); `small` = Hillstrom, RetailHero, Orange (1,620). |
| `recheck_ate_context.py` / `.csv` | Context check from the run records (3,900 runs): z = (predicted average effect - context difference in means) / SE. |
| `recheck_range.py` / `.csv` | Range check (3,900 runs): share of evaluation rows with abs(tau-hat) above 1 on a 0/1 outcome. |
| `recheck_holdout.py` / `.csv` | The same quantities on the saved Criteo holdout predictions (38 contexts). |
| `recheck_pilots.py` / `.csv` | Re-scores the three pilot files against the main mu0-hat. The source of every pilot number. |
| `recheck_summary.py` / `.txt` | Fold-level tables built from the CSVs above. |
| `recheck_figure.py` | Draft figure, reads `recheck_five_h18.csv` and `recheck_five_grouped.csv` only. |
| `recheck_cv.py`, `recheck_ate_sign.py`, `recheck_lzd.py`, `recheck_negated.py`, `recheck_shape.py` (+ `.csv`) | First-pass scripts on the paper's folds; superseded by `recheck_five.py` for the numbers quoted below. `recheck_shape.py` tests the "constant log-odds shift" reading, which is only partly supported. |
| `dopfn_flip_smoke.py` -> `dopfn_flip_smoke_2048.npz`, `dopfn_flip_smoke_fold3_seed44.npz` | CPU pilots: recoded outcome (`flip`) and relabelled treatment arms (`swap`). |
| `dopfn_norm_smoke.py` -> `dopfn_norm_smoke_2048.npz`, `dopfn_borders.py` | CPU pilot of the per-arm readouts and of overridden target statistics; bucket borders of the output head. The override is off the training distribution and is withdrawn as an experiment. |

Run from `repro/` (`.venv` for analysis, `.venv-dopfn` for the pilots, which set `CUDA_VISIBLE_DEVICES=""`):

    .venv/Scripts/python.exe -W ignore recheck_2026-11-10/recheck_five.py recheck_2026-11-10/recheck_five_h18.csv lzd,criteo home18
    .venv/Scripts/python.exe -W ignore recheck_2026-11-10/recheck_summary.py > recheck_2026-11-10/recheck_summary.txt
    .venv/Scripts/python.exe -W ignore recheck_2026-11-10/recheck_pilots.py recheck_2026-11-10/recheck_pilots.csv
    .venv/Scripts/python.exe -W ignore recheck_2026-11-10/recheck_figure.py
    .venv-dopfn/Scripts/python.exe -W ignore recheck_2026-11-10/dopfn_flip_smoke.py 8000 "T:swap,R:swap" recheck_2026-11-10/dopfn_flip_smoke_fold3_seed44.npz 3 44

## Results (Q-hat = normalized Qini; fold means; "risk" = same-rows risk-only ranking)

1. **The inversion holds.** Criteo, Do-PFN, natural contexts: Q-hat is negative in 5 of 5 folds at every K on both fold
   splits (paper's folds, K = 512 to 4,096: R -0.078, -0.074, -0.081, -0.083; T -0.072, -0.077, -0.085, -0.083; second
   split: R -0.094 to -0.083, T -0.052 to -0.091). Full holdout (2,093,918 rows, 3 contexts): T -0.046, T+Y +0.048.
2. **It is a risk ranking with the wrong sign.** All K, paper's folds: R -0.079 (between -0.100, within +0.031);
   T -0.079 (between -0.097, within -0.006); T+Y +0.051 (between +0.058, within +0.030). rho(tau-hat, mu0-hat) is
   -0.74 (R), -0.60 (T), +0.03 (T+Y). Under T+Y the score is nearly flat in risk: at K=1,024 the mean tau-hat is about
   +0.05 in the top two risk deciles and +0.01 in the bottom four (`recheck_shape.csv`).
3. **A risk-only ranking fitted on the same context rows beats every foundation model on Criteo.** It scores 0.082 to
   0.096 in all seven designs. Q-hat minus risk, all K, paper's folds: Do-PFN R -0.167 [-0.229, -0.106], T -0.162
   [-0.222, -0.103], T+Y -0.045 [-0.082, -0.007]; CausalPFN -0.057, -0.052, -0.080; TabPFN-T -0.066, -0.036, -0.070
   (intervals exclude zero in 4 of these 6 cells, and in 6 of 6 on the second split). T+Y is below risk in 5, 5, 4, 4 of
   5 folds at K = 512 to 4,096 (second split: 5, 5, 4, 5). The 20% design ties it: 0.087 against 0.092, -0.005
   [-0.043, +0.034]. Holdout: T+Y 0.048 against 0.077 (below in 3 of 3 contexts; 10 of 10 on the 200k subsample), 20%
   0.074 against 0.075 (below in 5 of 10).
4. **Negating the natural-context score does about as well as T+Y.** All K, paper's folds: negated R +0.074, negated
   T +0.074, T+Y +0.051; negated minus T+Y +0.023 [-0.006, +0.053] (R) and +0.023 [-0.005, +0.051] (T); negated minus
   risk -0.015 [-0.039, +0.010] and -0.009 [-0.045, +0.026].
5. **The sign is that of the predicted average effect, which is not anchored in the context.** Sign rule
   sign(rho) = sign(mean tau-hat) x sign(0.5 - context positive share): 178 of 180 Do-PFN runs on Criteo, 169 of 180 on
   LZD (T+Y excluded). Natural Criteo contexts, both splits, 240 runs: predicted average effect -0.183 against a context
   difference in means of +0.010; wrong-signed in 100% of runs; median 17 times too large and 12.6 SE away from the
   context's own contrast; 4.5% of rows have abs(tau-hat) above 1 (max 75.5) and every run has some. CausalPFN and
   TabPFN-T: 4.2% and 1.2% of runs beyond 2 SE, no row outside [-1, 1].
6. **Equal-share designs carry exactly zero treatment-outcome contrast in the context** (max abs 0.0000), and the
   predicted average effect takes a design-specific sign: Criteo 10% -0.033 (positive in 0 of 15 contexts), 20% +0.067
   (15), 35% -0.056 (0), T+Y +0.015 (13), 75% -0.069 (0). Not monotone in the share.
7. **Scope.** LZD shows the same signs within the noise floor (R negative in 5 of 5 folds on both splits; Q(mu0-hat)
   0.021 with SD 0.041 over the evaluation sets). Hillstrom, RetailHero and Orange show no inversion and no out-of-range
   value; on Hillstrom Do-PFN's average effect is right-signed (99.2% of natural runs) but 0.42 of the true size. The
   risk ranking is useless on these three (about zero), so point 3 is a Criteo statement.
8. **Checks without labels.** Range check (any abs(tau-hat) above 1): 100% of Do-PFN runs on Criteo, 95% on LZD, none
   for any other model or dataset. Context check (beyond 2 SE): 100% and 92.5%, but also 51.7% on Hillstrom and 25% on
   RetailHero, where nothing inverts: it detects an average effect that is not anchored in the context, not inversion.
9. **Pilots (single contexts, not experiments).** Criteo, K=1024, design T: relabelling the treatment arms reverses the
   predicted effect in 2 of 2 contexts. Fold 3, seed 44, all 8,000 rows: Q-hat -0.069 to +0.023, rho -0.57 to +0.66,
   mean tau-hat -0.044 to +0.077, rank correlation between the two scores -0.74; the arm coded 1 gets the lower
   prediction in 96.6% of rows before relabelling and 97.7% after. Under R (treated share 0.84 becomes 0.16 after
   relabelling) there is no mirror image. Recoding the outcome (one context, treated-only rows, no Q-hat) removes the
   out-of-range values and the wrong-signed average, and leaves rho near zero.
10. **Related work** (read on arXiv, see `PLAN_ru.md`, section 3.6). Do-PFN's authors report no real randomized trial
    and no binary outcome. CausalPFN's own uplift table includes Criteo but uses contexts of about 40,000 rows, a Qini
    normalised to the best model per column, and no risk-only reference; the paper does not mention that table yet.

## Caveats found on the way

- The two fold-0 pilot files hold the first 2,048 evaluation rows, which are all treated (the Criteo file is not
  shuffled: 12 treatment switches along 13,979,592 rows). No Qini there, and the rho printed by the pilot scripts used a
  risk model fitted on one low-base-rate segment. Use `recheck_pilots.csv`.
- The row order does not affect the main results: evaluation and holdout samples are uniform over the file, and HistGB
  predicts the treatment from the features with AUC 0.505. Whether the order comes from the public file or from the
  cleaning step was not determined.
- The CV run times in the records (median seconds per 8,000 rows at K=1024): Do-PFN 104 (Criteo) and 114 (LZD) in
  `home18`; CausalPFN 27; TabPFN-T 64. A CPU pilot cell takes 40 to 50 s on 12 threads. The
  DataSphere T4 records cover only 200,000-row runs (332 s of model time), so 1.5 min per 8,000-row run is an upper
  bound for planning.
