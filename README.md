# Outcome composition can invert Do-PFN's uplift ranking (Criteo)

Code, results and paper for a study of how the context rows of a causal foundation model are sampled.
Status on 10 November 2026: short paper prepared for WSDM 2027 (4 pages including references, ACM sigconf,
double-blind review). Do not link this repository from the submission.

**Main finding.** On the Criteo uplift benchmark, the released Do-PFN checkpoint ranks users in the wrong
direction when its 1,024-row context follows the natural visit rate (about 4%). Balancing outcomes within each
treatment arm (T+Y) reverses the sign without retraining: normalized Qini goes from −0.046 to +0.048 on the
2,093,918-row holdout. CausalPFN, a TabPFN T-learner and classical learners fitted on the same rows do not invert.

## Quick start: rebuild the figures and check the paper's numbers (CPU, about a minute)

Run from the repository root with Python 3.11. On Windows, clone into a short path such as `C:\src` (some result
paths are 170 characters long; Python cannot open paths beyond 260 characters unless long paths are enabled).

```bash
python3.11 -m venv .venv                                 # Windows: py -3.11 -m venv .venv
.venv/bin/pip install -r requirements-cpu.txt            # Windows: .venv\Scripts\pip ..., likewise below
.venv/bin/python paper/figures/make_figures.py           # rebuilds paper/figures/fig1-3 and their point tables
.venv/bin/python repro/scripts/paper_numbers.py          # 149 numbers: 145 compared with the paper, 4 reported
tectonic -X compile paper/main.tex                       # rebuilds paper/main.pdf
```

- `make_figures.py` and `paper_numbers.py` overwrite tracked files; `git status` shows whether anything changed.
  On a clean copy on the authors' Windows machine nothing did: the figures came out pixel-identical and all 145
  compared numbers matched.
- `paper_numbers.py` writes `repro/rerun_compare/paper_numbers.csv`, one row per number with the recomputed value,
  the value printed in the paper and the file it comes from, and exits with status 1 on a mismatch.
- The figures use Arial; without it matplotlib falls back to DejaVu Sans and the layout shifts slightly.
- [tectonic](https://tectonic-typesetting.github.io) (0.17 was used) is not installed by the steps above and
  downloads its TeX bundle on first use.
- The same environment runs the parts of the re-check that need no prediction arrays: `recheck_summary.py` (run
  inside `repro/recheck_2026-11-10/`), `recheck_figure.py` and `recheck_ate_context.py <output.csv>`. They also
  reproduce the committed outputs exactly.

## Layout

The repository keeps the project's own folder layout, so the analysis scripts run unchanged from the root.

| Path | What it holds |
|---|---|
| `paper/` | `main.tex`, `refs.bib`, `main.pdf`; `figures/make_figures.py` and the plotted points as CSV |
| `repro/rerun_compare/` | analysis tables behind the paper: `mechanism_runs.csv` (Criteo and LZD CV runs at K=1,024 with risk alignment and the same-row learners), `holdout_share_*.csv` (first holdout check of the share designs, 3 contexts), `hs10_*.csv` (10-context follow-up and its two-level bootstrap), `paper_extra_numbers.csv`, `paper_numbers.csv`, and two arrays for Fig. 3. The `CS*`, `M*`, `novelty_*` files and the `*_workflow.json` / `coherence_review.json` logs come from re-running the earlier HypeCheck study and from automated reviews of the drafts |
| `repro/home/projects/smiles/results/` | per-run JSONs: CV for all five datasets on scikit-learn 1.5.2 folds (`fm_cv`), the frozen full holdout (`fm_holdout`), the first share check (`fm_holdout_share`), TabPFN version scans (`scan*`) |
| `repro/home18/projects/smiles/results/` | per-run JSONs for Criteo and LZD on the scikit-learn 1.8 folds the paper uses: CV (`fm_cv`) and the outcome-share designs (`fm_dose`) |
| `repro/ds/results/hs_42` … `hs_51/` | the ten 200k-subsample holdout runs (20% and T+Y) from Yandex DataSphere: run JSONs and logs |
| `repro/home*/projects/smiles/manifests/`, `.../cache/*/*.json` | run manifests and the fold and cache metadata, with hashes to compare a rebuild against |
| `repro/home/projects/smiles/cpu_smoke/` | the original evaluation code (`fm_benchmark.py`, `fm_holdout.py`, ...); the Python files are unmodified |
| `repro/scripts/` | reproduction drivers and analysis scripts |
| `repro/recheck_2026-11-10/` | a re-check of the paper's claims made on 10 November; see its `README.md` |
| `repro/ds/` | DataSphere job tooling (`ds.py`, `job.sh`, Linux requirements) |
| `repro/patches/` | the 3-line Do-PFN change needed on torch 2.7 |
| `repro/env_main.txt`, `env_dopfn.txt`, `env_tab.txt`, `env.sh` | pinned environments and the shell setup the drivers source |
| `repro/holdout_share_protocol.md`, `repro/DEVIATIONS.md` | the protocol of the holdout share check; differences from the original report |
| `repro/e19_source_audit.json` | SHA-256 hashes of the cleaned data files |
| `repro/literature_review_2026-11-08/` | literature review notes and sources |
| `data_A_cleaned/*/cleaning_metadata.json` | how each dataset was cleaned; the data are not included |
| `docs/` | `DATA.md` (sources and preparation), `RERUN.md` (environments, checkpoints, rerunning a model), project notes |

## Where the paper's numbers come from

| Paper item | Computed by | Reads |
|---|---|---|
| Table 1, full holdout rows and T+Y − T | `repro/scripts/paper_numbers.py` | `repro/home/.../fm_holdout/criteo/holdout_dopfn_*_ctx1024_*.json` |
| Table 1, 200k-subsample rows | `hs10_analysis.py`, `holdout_share_analysis.py` (need prediction arrays) | their outputs `rerun_compare/hs10_runs.csv`, `holdout_share_runs.csv` |
| Fig. 1 | `paper/figures/make_figures.py` | `mechanism_runs.csv`, `holdout_share_runs.csv`, `hs10_runs.csv`, `hs10_contrasts.csv` |
| Fig. 2a, RQ1 context sizes | `make_figures.py`, `paper_numbers.py` | `repro/home18/.../fm_cv/criteo/fold*_dopfn_*.json` |
| Fig. 2b, RQ2 interaction, RQ3 CV contrasts | `make_figures.py`, `paper_numbers.py` | `mechanism_runs.csv` |
| Fig. 3 | `make_figures.py` | `rerun_compare/holdout_share_cache.npz`, `hs10_scores.npz` |
| Two-level bootstrap intervals | `hs10_analysis.py` (DataSphere job `hs10boot`) | `hs10_contrasts.csv` |
| Decile uplift, average effect, mean predictions | `paper_extra_numbers.py` (needs prediction arrays) | `paper_extra_numbers.csv` |
| `mechanism_runs.csv` itself | `repro/scripts/mechanism_probe.py` (needs prediction arrays and data) | |

The source column of `paper_numbers.csv` names the file behind each number.

## What is not included

- **Data.** See `docs/DATA.md`. Criteo is released under CC BY-NC-SA 4.0; files here that are derived from its
  rows (the run results and the two arrays in `repro/rerun_compare`) carry the same terms.
- **Model weights.** They are public; versions, download commands and checksums are in `docs/RERUN.md`.
- **Per-run prediction arrays** (`*.npz`, about 4 GB). They are needed to recompute `mechanism_runs.csv`, the
  subsample analyses, the bootstrap and most of the re-check from scratch. Not published yet.
- **The data-cleaning script** (`code/prepare_benchmark_data.py`) is not part of this project; each
  `cleaning_metadata.json` lists the steps it applied.
- **The HypeCheck reproduction trees and the original report.** `repro/SMILES-HypeCheck`, `repro/rerun`,
  `repro/archived_from_rerun` and `report_FM.docx` are not included. Scripts that rebuild or compare them do not
  run from this repository: `adopt.py`, `adopt_runs.py`, `night_gpu.sh`, `night_extra.sh`, `overnight.sh`,
  `queue.sh`, `compare_cs01.py`, `compare_report*.py`, `novelty_a.py`, `cpu_speed_e17.py`, `lzd_prior_ids.py`,
  `trace_inputs.py`, and the input packs of `repro/ds/ds.py`.
- Local paths in logs and notes are replaced by `<project>`, `<home>`, `<scratch>` or `<session>`, cluster user
  folders by `<user>`, cloud resource ids by `<id>`.

Differences from the original report (fold versions, TabPFN-T, the Do-PFN patch) are listed in
`repro/DEVIATIONS.md`.

## Licence

Not chosen yet; ask the authors before reusing the code. Criteo-derived files: CC BY-NC-SA 4.0.
