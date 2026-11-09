# Holdout check of the outcome share: protocol (fixed 2026-11-09 18:40, before any run)

> **Correction added after the runs (19:40).** The time in the title is wrong. The file was created at 18:36:16, and the first runs started at 18:37:05 (`created_utc` 15:37:06Z). Results were appended at 19:03 and no hash of the pre-run text was recorded. Describe this check as specified in an internal protocol before results were available, not as pre-registered.

**Question.** Does the outcome share selected on CV (20% of each treatment arm with Y=1) beat T+Y (50%) and T (the natural share) for Do-PFN on the Criteo holdout?

## Design

- Model and data: Do-PFN, Criteo-ITE-v2.1 (outcome visit), K = 1024.
- Context: drawn from the development rows only, exactly as `fm_holdout.py` does (`choose_context(dev, ..., fold=0, context_seed=seed)`), for seeds 42, 43 and 44.
- Strategies:
  - `share20` is the primary new run; `share35` is secondary and tests the CV dip.
  - T (`balanced_treatment`) and T+Y (`balanced_treatment_outcome`) are not rerun. Their existing full-holdout predictions from the frozen 48-run holdout (same contexts, same seeds) are restricted to the same evaluation rows.
- Evaluation rows: the fixed holdout subsample from `fm_holdout.choose_eval(final_idx, t, y, 200000, fold=0)`, identical for every strategy and seed.
- Runner: unmodified `fm_holdout.py`, with the shareNN sampler added by `scripts/run_dose_holdout.py`. The sampler is the same as in `scripts/run_dose_cv.py`. HOME is `repro/home`, as for the original holdout runs; outputs go to `results/fm_holdout_share` (the frozen holdout folder is not touched).

## Endpoints

- Primary: normalized Qini (Q-hat), the mean of the three seeds.
- Secondary:
  - normalized AUUC and uplift@10%;
  - Spearman correlation of the score with baseline risk mu0(x) and with an independent uplift estimate tau(x), both from a HistGB T-learner fitted on development rows only;
  - Q-hat of big-data S- and T-learners fitted on development rows and scored on the same evaluation rows.

## Hypotheses (signs taken from CV, fixed now)

- H1: Q-hat(20%) > Q-hat(T+Y). CV: +0.032, positive in 5 of 5 folds.
- H2: Q-hat(20%) > Q-hat(T). CV: +0.164.
- H3: Q-hat(35%) < Q-hat(20%). CV: −0.066, 13 of 15 runs.
- H4: rho(score, mu0) > 0 at 20%, and rho(score, mu0) < 0 at T and at 35%.

## Decision rule (per hypothesis)

A hypothesis counts as replicated only if all three hold:
- the mean difference over the three seeds has the CV sign;
- at least 2 of 3 seeds have that sign;
- the paired cluster-bootstrap 95% interval of the seed-averaged difference excludes zero. The bootstrap uses 1000 draws over clusters of identical feature vectors, with the same draws for every strategy and seed.

Otherwise the hypothesis is reported as not replicated. All results are reported either way, together with three-seed t-intervals.

## Results (added after the runs, 2026-11-09 19:00)

The runs are six Do-PFN holdout runs (share20 and share35 × seeds 42–44) on the local RTX 5090, with the user's approval, at about 8 minutes each; all ended with rc=0. Evaluation used the fixed 200k holdout subsample (eval hash cf7d585…; ATE +1.01 pp). The T and T+Y scores are the frozen full-holdout predictions restricted to the same rows; the T+Y context hash equals the frozen run's. The bootstrap uses 1000 cluster draws in 10 independent parts. Tables are in `rerun_compare/holdout_share_runs.csv` and `holdout_share_contrasts.csv`.

| Strategy | Q-hat by seed (42 / 43 / 44) | uplift@10%, pp | rho(score, mu0) |
|---|---|---|---|
| T | −0.045 / −0.052 / −0.062 | 2.0 / 1.7 / 0.7 | −0.34 / −0.62 / −0.50 |
| 20% | 0.068 / 0.079 / 0.078 | 5.15 / 5.37 / 5.16 | +0.20 / +0.64 / +0.22 |
| 35% | −0.008 / −0.014 / 0.000 | 1.5 / 2.4 / 2.5 | −0.57 / −0.46 / −0.36 |
| T+Y (50%) | 0.032 / 0.055 / 0.065 | 2.8 / 4.1 / 3.4 | −0.12 / −0.03 / +0.08 |
| Reference S-learner (2M development rows) | 0.082 | 5.36 | +0.79 (tau versus mu0) |
| Reference T-learner (2M development rows) | 0.079 | 5.28 | |

Hypotheses (decision rule from the protocol):

- H1, 20% − T+Y: Q-hat +0.0245, bootstrap [+0.004, +0.044], 3 of 3 seeds positive (t-interval [−0.003, +0.052]). **Replicated.** For uplift@10% it is +1.80 pp [+0.64, +2.95].
- H2, 20% − T: +0.128 [+0.091, +0.168], 3 of 3 seeds. **Replicated.**
- H3, 35% − 20%: −0.082 [−0.110, −0.055], 3 of 3 seeds negative. **Replicated.**
- H4: rho(score, mu0) is positive at 20% and negative at T and at 35% in every seed. ~~Replicated.~~ **Not formally replicated:** the protocol's bootstrap interval for rho was not computed, so only the seed-sign condition is met.

Corrections from the independent verification (`rerun_compare/verify_rq_workflow.json`):

- **The intervals understate uncertainty.**
  - The cluster bootstrap holds the three contexts fixed. The 200k subsample has 197,021 clusters, so it is close to a row bootstrap.
  - For H1 the three-seed t-interval includes zero. Across four hypotheses Bonferroni would lose H1, while Holm keeps it.
  - A positive claim that 20% beats T+Y needs more context seeds, T+Y re-scored on the 200k rows, and a two-level interval.
- **The CV figures in the hypotheses round to +0.031 (H1) and +0.165 (H2),** not +0.032 and +0.164.
- **The holdout is not untouched.** It was scored in the 48 frozen runs, and the CV mechanism probe fitted its Criteo reference on holdout rows.
- **For Do-PFN the run seed changes only the context.** The model's internal RNG is fixed (seed 0) and redraws polynomial features on every predict call; no batch-layout effect was found.

## Follow-up: 10 context seeds on T4 (DataSphere, 9 November 2026)

Added after the three-seed check above, to settle H1. Seeds 42–51 for 20% and T+Y were scored on the same 200k rows on DataSphere T4 GPUs (jobs `hs_42`…`hs_51`). Seeds 42–44 repeat the first check on a different GPU and agree with it within 0.001 in Q-hat. Analysis: `scripts/hs10_analysis.py`; outputs `rerun_compare/hs10_runs.csv`, `hs10_seed_summary.csv` and `hs10_contrasts.csv` (job `hs10boot`).

| Contrast | Mean | t-interval over 10 contexts | Two-level bootstrap (contexts × row clusters, 992 draws) |
|---|---|---|---|
| 20% − T+Y, Q-hat | +0.0245, 10 of 10 positive | [+0.015, +0.034] | [+0.005, +0.043], P(≤0) = 0.010 |
| 20% − T+Y, uplift@10%, pp | +1.17, 9 of 10 positive | [+0.32, +2.02] | [−0.15, +2.39] |
| 20% − S-learner reference, Q-hat | −0.008 | [−0.013, −0.003] | [−0.022, +0.006] |

Reading: the Q-hat advantage of 20% over T+Y holds when both contexts and rows are resampled. The uplift@10% advantage and the gap to the S-learner reference are not resolved once rows are resampled. The comparison was chosen after the three-seed check, so this is a follow-up with more seeds, not an independent replication.
