# Rerunning the models

The runs behind the paper used Python 3.11.9 locally (RTX 5090, Windows, Git Bash) and Python 3.11.13 on Yandex
DataSphere (Linux; the 10-context holdout runs on a Tesla T4). Model runs need the data (`docs/DATA.md`) and the
checkpoints below. The recipes on this page follow how the runs were made; they were not re-run from a fresh
clone of this repository.

## Environments

| Venv (path the drivers expect) | Pinned list | Used for |
|---|---|---|
| `repro/.venv` | `repro/env_main.txt` | CausalPFN, TabPFN T-learner, analysis scripts, figures |
| `repro/.venv-dopfn` | `repro/env_dopfn.txt` | Do-PFN only (it needs pandas < 2.2) |
| `repro/.venv-tab` | `repro/env_tab.txt` | building the Criteo and LZD fold caches with scikit-learn 1.8.0, which the paper's folds need |
| any | `requirements-cpu.txt` | `paper_numbers.py`, `make_figures.py` and the CSV-only re-check scripts |

Linux / CUDA 12.6 versions of the first two lists are `repro/ds/requirements-main.txt`, `requirements-dopfn.txt`
and `linux-extras.txt`. Install the `env_*.txt` and `repro/ds` lists with `pip install --no-deps -r <file>` (the
CUDA 12.8 torch wheel comes from `--extra-index-url https://download.pytorch.org/whl/cu128`); without `--no-deps`,
kditransform pulls scikit-learn >= 1.6 and moves it off 1.5.2. Install `requirements-cpu.txt` with plain pip.
The drivers look for `Scripts/python.exe` (Windows) inside these venvs; `repro/env.sh` falls back to `bin/python`.

## Checkpoints

| Model | Source | Version | File | SHA-256 |
|---|---|---|---|---|
| Do-PFN | `https://github.com/jr2021/Do-PFN.git` | commit `90d67433b43c4d52d752dc336070f525ff856e0b` | `artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt` | `630480c82e33fcada67666833b532f4483987c804124a685d3b20c77a5d45d03` |
| | | | `artifacts/dopfn_model.pkl` | `6e05434bedcbec8c7a8729fa3ccb42e0a37788e2d055bd9c6879f6ef2ae36e2b` |
| | | | `artifacts/dopfn_config.pkl` | `e8607e55bda3251289ea041c3277d50f6fa36f6183b92ad17fa4218cac851ed3` |
| CausalPFN | Hugging Face `vdblm/causalpfn` (package `causalpfn==0.1.5`) | revision `83aad07da1cb077cfda4236878a1b07dc9f72a54` | `causalpfn_v0.pt` | `4f0f53717f1c62426fe98faa0459d33bc8f9862a87e8484ca3869ca899e7841d` |
| TabPFN | Hugging Face `Prior-Labs/tabpfn_2_6` (package `tabpfn==9.2.0`) | revision `24148873a9a5992b429833d07363690dcffd02a8` | `tabpfn-v2.6-classifier-v2.6_default.ckpt` | `0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a` |

The Do-PFN weights are committed in its repository. The other two are downloaded into the places the evaluation
code reads (`H` is the `HOME` of the run, see below; `rerun_env.sh` sets `HF_HUB_OFFLINE=1`, so download first):

```bash
huggingface-cli download vdblm/causalpfn causalpfn_v0.pt --revision 83aad07da1cb077cfda4236878a1b07dc9f72a54 \
    --cache-dir "$H/.cache/causalpfn"
huggingface-cli download Prior-Labs/tabpfn_2_6 tabpfn-v2.6-classifier-v2.6_default.ckpt \
    --revision 24148873a9a5992b429833d07363690dcffd02a8 --local-dir "$H/projects/smiles/models/tabpfn"
sha256sum "$H"/.cache/causalpfn/models--vdblm--causalpfn/snapshots/*/causalpfn_v0.pt \
    "$H"/projects/smiles/models/tabpfn/tabpfn-v2.6-classifier-v2.6_default.ckpt
```

TabPFN 2.6 weights have their own licence (TABPFN-2.6 License v1.0, in the Hugging Face snapshot).

## One Do-PFN run (Criteo, fold 0, T, K=1,024, seed 42)

The original evaluation code (`repro/home/projects/smiles/cpu_smoke/`) finds data, caches, models and results under
`~/projects/`, so each run gets its own `HOME`. Use a fresh one: `repro/home` and `repro/home18` hold the shipped
results and cache metadata, which the cache builders refuse to overwrite and the run would overwrite.

```bash
cd repro && source env.sh                         # sets R (= repro/), CODE, winpath
H="$R/rerun_home"; mkdir -p "$H/projects/smiles"
ln -s "$(cd "$R/.." && pwd)" "$H/projects/smiles/data"      # a folder that holds data_A_cleaned/
git clone https://github.com/jr2021/Do-PFN.git "$H/projects/Do-PFN"
git -C "$H/projects/Do-PFN" checkout 90d67433b43c4d52d752dc336070f525ff856e0b
git -C "$H/projects/Do-PFN" apply "$R/patches/dopfn_torch27_layer.patch"   # needed on torch 2.7
export HOME="$H" USERPROFILE="$(winpath "$H")"
cd "$CODE"
"$R/.venv-tab/Scripts/python.exe" make_group_split_cache.py --dataset criteo  # scikit-learn 1.8.0
"$R/.venv-tab/Scripts/python.exe" make_dense_cache.py
"$R/.venv-dopfn/Scripts/python.exe" fm_benchmark.py --dataset criteo --model dopfn --fold 0 --seed 42 \
    --context-size 1024 --strategy balanced_treatment --eval-max-size 8000 --device cuda:0
```

- On Windows, `ln -s` in Git Bash copies instead of linking; use `cmd //c mklink /J <link> <target>` with Windows
  paths, or set `MSYS=winsymlinks:nativestrict` (Developer Mode).
- Check the fold cache first: `labels_sha256`, `final_holdout_hash` and `group_hash_sha256` in
  `$H/projects/smiles/cache/group_splits/criteo_fold_membership.json` must equal those in
  `repro/home18/projects/smiles/cache/group_splits/`; `created_utc` and file modification times will differ.
- Then compare `$H/projects/smiles/results/fm_cv/criteo/fold0_dopfn_balanced_treatment_ctx1024_seed42_eval8000_cuda0.json`
  with the shipped file of the same name under `repro/home18/projects/smiles/results/fm_cv/criteo/`:
  `split_cache_sha256`, `context_hash` and `eval_hash` must match exactly; the metrics agree to fp16 precision
  (`repro/DEVIATIONS.md`). The run took about 94 s on the RTX 5090. `--preview --device cpu` is a dry run
  without the model.

## Batches, the holdout and the share designs

The drivers in `repro/scripts` are the records of how the batches were launched, written for Git Bash on
Windows; read them for the environment variables each step needs.

- CV batches: `repro/scripts/run_cv.py <manifest>` with the manifests in `repro/home*/projects/smiles/manifests/`
  (`rerun_folds.sh` for the paper's Criteo and LZD folds).
- Frozen full holdout: `run_holdout.py <manifest> <workers>`, called from `overnight.sh` with
  `home/projects/smiles/manifests/holdout_frozen_20261021_233147/frozen_holdout_manifest.txt`, whose SHA-256 is
  recorded in every holdout JSON; it needs `R_WIN` and `CODE_WIN` (set in `overnight.sh`) and an existing
  `home/projects/smiles/logs/`.
- Share designs: `run_dose_cv.py` (needs `FM_OUTPUT`) and `run_dose_holdout.py` (needs `HOLDOUT_OUT`); see the
  `dose` and `holdout_share` cases in `repro/ds/job.sh` and `scripts/holdout_share.sh`. Protocol:
  `repro/holdout_share_protocol.md`.
- The 10-context follow-up ran on DataSphere (`repro/ds/ds.py`: jobs `hs_42` ... `hs_51`, then
  `scripts/hs10_analysis.py collect` and the CPU bootstrap job `hs10boot`). As shipped, `ds.py packs` cannot
  build its input packs, because they include trees and logs that are not in this repository; the DataSphere
  tooling is a record of what was run. It also needs a `repro/.venv-ds` with the DataSphere CLI, your own project
  (`DS_PROJECT`) and a service-account key (`YC_SA_KEY_FILE`).
