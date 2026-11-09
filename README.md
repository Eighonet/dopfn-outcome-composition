# Outcome composition can invert Do-PFN's uplift ranking (Criteo)

Results, code and paper for the SMILES study of how the context of a causal foundation model is sampled.
Status on 10 November 2026: WSDM 2027 short-paper draft (4 pages, ACM sigconf, double-blind).

**Main finding.** On the Criteo uplift benchmark, the released Do-PFN checkpoint ranks users in the wrong
direction when its 1,024-row context follows the natural visit rate (about 4%). Balancing outcomes within each
treatment arm (T+Y) reverses the sign without retraining: normalized Qini goes from −0.046 to +0.048 on the
2,093,918-row holdout. CausalPFN, a TabPFN T-learner and classical learners fitted on the same rows do not invert.

## Layout

| Path | What it holds |
|---|---|
| `paper/` | `main.tex`, `refs.bib`, compiled `main.pdf`; `figures/make_figures.py` rebuilds every figure from saved results, with the plotted points as CSV |
| `docs/` | research questions and answers, review of the student drafts, the hand-off note, the holdout protocol, the log of deviations from the original report, the literature review |
| `results/rerun_compare/` | analysis tables: mechanism probe, outcome-share dose curves, holdout share check, 10-seed T4 follow-up, two-level bootstrap, independent verification outputs |
| `results/home/`, `results/home18/` | per-run result JSONs of the reproduced report (CV, frozen holdout, share runs, dose runs). For Criteo and LZD CV use `home18` (the folds that match the report) |
| `repro/scripts/` | reproduction and analysis scripts |
| `repro/ds/` | Yandex DataSphere job tooling (`ds.py`, `job.sh`, requirements). Rendered configs, input packs and the project id are not included |
| `repro/report_code/cpu_smoke/` | the report's evaluation code that the scripts import (metrics, manifests) |

## Not included

Raw data (Criteo, LZD and others), model weights, virtual environments, prediction arrays (`*.npz`), DataSphere
input packs, chat exports and any credentials. Paths in the scripts assume the original project layout
(`causal_fm_project/repro/...`).

## Rebuilding the paper

```bash
python paper/figures/make_figures.py
tectonic -X compile paper/main.tex
```

`make_figures.py` expects the original layout, where `rerun_compare` and the `home*/projects/smiles/results`
folders sit under `repro/`.
