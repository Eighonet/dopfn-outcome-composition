# Data

The data are not in this repository. The code expects a folder `data_A_cleaned/` next to `repro/` with one
sub-folder per dataset, each holding two Parquet files joined on `epk_id`:

- `features.parquet`: `epk_id`, `T` (treatment, 0/1), `treatment_dt`, then the pre-treatment features. Every
  column other than `epk_id`, `T`, `treatment_dt` and `lag` is used as a feature, so nothing else may be in it.
- `outcomes.parquet`: `epk_id`, `lag` (absent for RetailHero) and the outcome columns (Criteo: `visit`,
  `conversion`, `exposure`; Hillstrom: `visit`, `conversion`, `spend`; LZD and RetailHero: `Y`; Orange: `churn`).

See `repro/home/projects/smiles/cpu_smoke/fm_benchmark.py`, lines 74-144.

The cleaned copies were made by a script that is not part of this project (`code/prepare_benchmark_data.py`,
2026-07-09). Its steps are recorded in each `data_A_cleaned/<dataset>/cleaning_metadata.json`, which is included.
`repro/scripts/verify_data.py` checks a local copy against the SHA-256 hashes in `repro/e19_source_audit.json`.
Row order and `epk_id` drive the group splits and every recorded hash, so an exact reproduction needs the
authors' cleaned files; ask the authors for them. A copy rebuilt from the public sources will give other folds.

| Folder | Rows | Features used | Outcome | Public source | Cleaning |
|---|---|---|---|---|---|
| `Criteo-ITE-v2.1` | 13,979,592 | 10 (f0-f11 without f7, f10) | `visit` | Criteo AI Lab, [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), v2.1 (Diemert et al., arXiv 2111.10106); also `sklift.datasets.fetch_criteo`. Licence CC BY-NC-SA 4.0 | f7 and f10 dropped as near-duplicates (abs. Spearman rho >= 0.99 with another feature); duplicate feature vectors (about 15% of rows) kept and handled by group splits |
| `LZD` | 181,669 | 64 | `Y` | Lazada data released with DESCN (Zhong et al., KDD 2022) | test split only (the randomized one); 19 near-duplicate features dropped (rho >= 0.99) |
| `Hillstrom` | 64,000 | 8 | `visit` | Kevin Hillstrom's MineThatData e-mail challenge (2008); `sklift.datasets.fetch_hillstrom` | none; `T` = 1 for either e-mail arm (42,694 treated, 21,306 controls) |
| `Retailhero-uplift` | 200,039 | 5 | `Y` | X5 RetailHero uplift competition; `sklift.datasets.fetch_x5` | labelled train split only; `first_redeem_time` and `issue_redeem_delay` dropped as partly post-treatment |
| `Orange Telecom Churn` | 11,896 | 178 (PC1-PC160, FACTOR1-FACTOR18) | `churn`, used as retention `1 - churn` | not recorded in this project | none |

Hashes of the Orange files, which `e19_source_audit.json` does not cover: `features.parquet`
`e51142cf026361031e65a5edb22ea4d5d4a0092f1e12debcf8570be7bab0f60b`, `outcomes.parquet`
`083832bb801ac137ce5b3dd1760a578050fcb1682540efefbefbbd3df48d57a4`.

## Splits

Constants in `fm_benchmark.py`: final holdout 15% (seed 42), 5 outer folds (seed 42), 8,000 evaluation rows per
fold (stratified, seed 42 + fold), contexts drawn with seed + fold. Strata are 2T + Y. Criteo and LZD are split by
groups of identical feature vectors (`GroupShuffleSplit`, `StratifiedGroupKFold`) and read the splits from a cache
built by `make_group_split_cache.py`.

- The Criteo holdout has 2,093,918 rows; its hash is in every holdout JSON (`full_final_holdout_hash`) and in
  `repro/home18/projects/smiles/cache/group_splits/criteo_fold_membership.json`.
- The paper's Criteo and LZD folds come from scikit-learn 1.8.0. Versions 1.5.2-1.7.2 give different fold labels
  (the holdout does not change); `repro/home/` holds that second split, `repro/home18/` the paper's.
  See `repro/DEVIATIONS.md`.
- The 200k-row holdout subsample is fixed by `choose_eval(final_idx, t, y, 200000, fold=0)`; its hash,
  `cf7d585ab7d1bb46ef6714eab3758e4232b7034bd0dc68812de1b39cf4288bdb`, is the `eval_hash` of every `eval200000` run
  JSON.

## Rows that this repository does contain

The run JSONs hold only metrics, hashes and counts. Two arrays in `repro/rerun_compare/` hold row-level values for
the 200k rows of the Criteo holdout subsample:

- `holdout_share_cache.npz`: `ev` (row indices into the cleaned Criteo file), `y` (visit), `t` (treatment),
  `mu0`, `mu1` (estimated visit probability without and with treatment: the two arms of the HistGB T-learner
  fitted on 2M development rows), `tau_t = mu1 - mu0`, `tau_s` (the HistGB S-learner's effect estimate), `cluster` (group of identical feature vectors), and
  Do-PFN scores `score_T_42..44`, `score_TY_42..44`;
- `hs10_scores.npz`: `y`, `t`, `cluster`, `tau_s`, `tau_t` and Do-PFN scores `20%_42..51`, `TY_42..51`.

Being derived from Criteo, they are under CC BY-NC-SA 4.0.
