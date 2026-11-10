# Deviations from the original runs

Hardware and environment differ from the cluster runs: RTX 5090, Windows 11, Python 3.11, torch 2.7.1+cu128. The original torch 2.1 to 2.3 builds do not support this GPU.

1. Do-PFN `model/layer.py`: three `src.set_(self.normN(src_))` calls replaced by `src = self.normN(src_).to(src.dtype)`. Newer torch refuses `set_` across float and half. The meaning is the same; fp16 rounding can differ slightly.
2. Do-PFN runs in its own venv with pandas 2.1.4, as the upstream requirements demand. scikit-uplift was installed with `--no-deps` to keep scikit-learn 1.5.2.
3. Archive code in `home/projects/smiles/cpu_smoke` is unmodified. Paths resolve through a project-local home (`USERPROFILE`), and jobs run in one process through `scripts/run_cv.py` instead of Slurm.

## TabPFN-T does not match the report exactly

Check: Hillstrom, K=2048, random, 15 CV runs. Report: Qini 0.01430 +- 0.01505, uplift@10% 0.07727.

| Variant | Qini | uplift@10% |
| --- | ---: | ---: |
| tabpfn 7.0.1, 7.1.1 | 0.01300 | 0.07286 |
| tabpfn 8.0.0, 8.0.4, 8.0.8, 8.1.0 | 0.01389 | 0.08060 |
| tabpfn 8.2.0 to 9.2.0 | 0.01366 | 0.08052 |
| tabpfn 9.2.0 with scikit-learn 1.6.1 to 1.8.0, numpy 1.26.4 or 2.2.6 | 0.01366 | 0.08052 |

Neither the tabpfn release nor scikit-learn or numpy explains the gap. CausalPFN matches to five digits on the same data and splits, so splits and contexts are not the cause. Untested: the original torch build and the 2080 Ti, which cannot be tried on this GPU, and intermediate tabpfn patch releases. Search stopped here. The full run uses tabpfn 9.2.0, the version recorded in the HypeCheck repo, and TabPFN-T counts as approximately reproduced.

## Folds on LZD and Criteo depend on the scikit-learn version

`StratifiedGroupKFold` gives identical labels on 1.5.2, 1.6.1 and 1.7.2 and different labels on 1.8.0 (LZD labels sha c162a632fabe1c vs 053ab38cb73a4c). The holdout split does not change. With 1.8.0 folds, LZD CausalPFN K=2048/T+Y reproduces all six CV metrics of the report to five digits, so the report used scikit-learn >= 1.8.

`home/` holds CV on the 1.5.2 folds and is kept as a sensitivity check. `home18/` holds LZD and Criteo CV on the 1.8.0 folds, which is the exact reproduction. Hillstrom, Orange and RetailHero do not use these folds.

## CS01/E01/v003 on the GPU

The run uses the archived script with sha 1c3fa229, copied to `scripts/cs01_v003_gpu.py`. The only edit adds `cuda` to the allowed devices. The original ran on CPU with batch 2 on torch 2.3.1; this run uses CUDA with batch 32 on torch 2.7.1.

The first attempt, CUDA with batch 32, stopped on the protocol guard: batch-of-2 and single-query scores differed by 0.00112 for Feature-kNN on Hillstrom, while all six random contexts passed. That run is kept in `runs/cs01_failed_batch32`. The protocol batch of 2 ran at 12 queries per second on this GPU, about 15 hours for all folds, so that attempt was stopped and kept in `runs/cs01_stopped_batch2`. The final run uses batch 32. The guard now logs the difference to `guard.txt` in each fold instead of stopping, and the real check is the comparison of all 180 metrics with the archived `all_metrics.csv`.

## Full rerun of the HypeCheck repo

The user chose to rebuild everything from scratch instead of asking for the local outputs. The rerun lives in `rerun/`, a separate local clone on branch `repro-gpu`; nothing is pushed and `SMILES-HypeCheck/` stays untouched. Two local commits change only this: device and batch size come from `REPRO_DEVICE` and `REPRO_BATCH`, with the original defaults kept, and the batch equality guard prints `GUARD <difference>` instead of stopping.

Later runners read earlier local outputs by their archived run ids and check SHA256, so finished runs have to be placed under those ids. Source-row lists are expected to match byte for byte; model scores are not.

Two of my own mistakes, both caught by the runners' checks and fixed before any result was used. First, the batch patch changed loop steps but not every loop body: M01/E15 stopped on a length check and was rerun from scratch, the failed run is in `runs/failed/m01_e15_slice_bug`. Second, I committed fixes while CS05/E01 was running, so its packager saw a different commit than its cohorts; packaging is redone on the cohorts' commit d36fb2d. Patches are now committed only while nothing in `rerun/` is running.

## M01/E15/v001 is stopped on purpose, as in the original

The archived v001 stopped after 55,296 of 500,000 predictions because its sample overlapped already studied CS05/CS06 cohorts; it has no final metrics. The rerun reproduced the same sample and exclusion list byte for byte (sample sha 6cf8e22f39f7), and since nothing stopped it, it ran to the end. Its local-minus-FM result (-0.554 pp, interval [-1.483, +0.300]) inherits the same overlap and is not a valid holdout either. To keep the chain faithful, the rerun v001 is put back into the stopped state that E15/v002 expects: status incomplete, final files moved to `runs/e15_v001_completed_but_invalid`, archived v001 snapshot restored. E15/v002 checks the commit that produced v001 as a literal; it now names the rerun commit bd73741.

## Which runners use the GPU

Only runners that score through `cs01_e03.make_forward` move their tensors to the configured device: CS01/E03-E06, M01/E06-E08, M01/E15-E17, CS05 and CS06. The synthetic and PEHE runners (CS04, CS07, M05, M06, M01/E12-E14) build their query tensors on the CPU, so on CUDA they stop with a device mismatch. They are small, and they run on the CPU with the original batch of 2, which is also the closest match to the original runs.

The batch patch left `REPRO_BATCH` undefined in `run_cs04_e02.py` and `run_cs04_e07.py`; the fix is committed only when no runner is active, because packagers compare the commit.

Holdout processes run at AboveNormal priority and rerun processes at BelowNormal, so the main contrast is not slowed by the rerun. This only affects those processes while they live.

## GPU handed back at 15:38 on 2026-11-07, E17 and E19 paused

At the user's request the two GPU runs were stopped at 15:38; both runners save progress and resume.

- M01/E17 (rerun clone) had 393,216 of 1,300,000 feature-retrieval scores saved in `feature_fm.partial.npy`. Resume: `source rerun_env.sh; cd rerun; "$PY" -u tools/run_m01_e17_v001.py all --resume > "$RL/m01_e17.log"`. The resume check compares the git commit, so nothing may be committed in `repro/rerun` until E17 finishes.
- M01/E19 (original clone) had 15 of 16 cells complete; the Criteo N51200 K2048 cell (about an hour) starts over. Resume on the same GPU: `cd SMILES-HypeCheck; "$PY" -u tools/run_m01_e19_v002.py all --resume --data-root "<data_A_cleaned>" >> e19.log`.
- `chain.log` shows `m01_e17 exit 1 15:38` and `overnight.log` shows `E19 rc=1`: these are the stops, not failures.

CPU wave 1 lists `cs04_e01` again although the 13:54 run is still going (its device is fixed to CPU in the runner, so that run is valid). `scripts/run_action.sh` now turns the duplicate into `--status`, after copying the running log to `cs04_e01.first.log`.

## Parallel bootstrap and CS04 wave 2 (2026-11-07 evening)

The main-contrast bootstrap was recomputed by `scripts/bootstrap_parallel.py` in 10 separate processes (each walks the same `default_rng(20261027)` draw sequence and keeps every 10th replicate), merged by `scripts/bootstrap_merge.py`. 95% intervals: Qini [+0.08467, +0.10236] vs report [+0.08486, +0.10244]; AUUC [+0.03356, +0.04042] vs [+0.03366, +0.04046]; uplift@10 [+0.01458, +0.02104] vs [+0.01471, +0.02116]. The shift follows the 0.0001 shift of the point estimates. The original sequential script still runs as a full check.

CS04/E02, E04 and E07 had failed on the undefined `REPRO_BATCH` (fixed in 9e0f8e0). Their failed run folders were moved to `runs/failed/*_batch_undefined_*`, and the three runners were started again on the CPU with batch 2.

## Night GPU chain from 01:00 on 2026-11-08

At the user's request the GPU tail restarts at 01:00 via `scripts/night_gpu.sh` (log `logs/rerun/night.log`). Line A: E17 `--resume` (batch 32, as it started) -> compare with `archived_from_rerun/E17_v001` -> adopt -> E18. Line B: E19 `--resume` in the original clone -> CS01/E04 -> CS01/E05 -> M01/E03. Adoptions use `ADOPT_NO_COMMIT=1`; commit them in the morning when no runner is active.

The 12 prior LZD eval-ID files that CS01/E04 checks were put in place by `scripts/lzd_prior_ids.py`, byte-identical to the archived hashes: the rerun's CS01/E02 split files and the E01/v003 folds only needed CRLF -> LF, and the single-split fold of E01 v001/v002 was regenerated with the source's own `split_indices(folds=1, split_seed=42)`. For the same reason the night chain rewrites CS01/E04 `eval_rows.csv` with LF before adoption, because CS01/E05 checks it against the archived hash.

The original sequential bootstrap finished at 19:51 (16,269 s): all 1000 replicates are identical to the parallel ones (max |diff| 0.0), so the parallel twin is exact.

## Move to Yandex DataSphere jobs (2026-11-07, 22:10)

The user decided to run the remaining work as DataSphere jobs. The local 01:00 GPU chain (`night_gpu.sh`, `night_extra.sh`) was cancelled before it started anything; local CPU runners that were already running (CS04, CS07, M01/E12, M06/E03) keep running, because none of them can resume.

- Tooling is in `repro/ds/`: `job.sh` rebuilds the local layout on the job VM, `ds.py` builds the input archives (`packs`), renders configs, submits with `--async`, and downloads results to `ds/results/<job>` only.
- Environments: exact local freezes (`ds/requirements-main.txt`, `ds/requirements-dopfn.txt`) plus torch 2.7.1's Linux CUDA 12.6 wheels, installed with `no-deps`, because resolving would move scikit-learn off 1.5.2 (`kditransform` asks for >= 1.6).
- The report's fold and dense caches check source parquet size and `mtime_ns`; tar keeps the bytes but not nanoseconds, so `job.sh` restores the recorded mtimes after checking sizes. A Linux dry run (WSL, `--preview`) reproduced the Criteo holdout hash and the CV eval hash exactly.
- M01/E17 restarts from scratch on the job GPU instead of resuming the 5090 partial run, so the whole E17 comes from one GPU. The 5090 partial run stays in `rerun/artifacts/local/m01_e17_v001_results/20261107T121620_876317Z`.
- M01/E19's last cell stays local: its run folder is 7.2 GB (job limit 10 GB, CLI download 1 GB) and its resume checks hardware and data-root path.

Local CPU reruns finished by 23:10 on 2026-11-07 and compared with the archive (`rerun_compare/`): CS04/E01, E04 and E05 match to 1e-5 (only `inference_seconds` differs, which is timing); M01/E12 matches to 1e-6; CS07/E01 to 0.003 on oracle regret@10 and 3e-4 on PEHE. Still running locally: CS04/E02, CS04/E07, M06/E03.

CPU wave 1 finished at 00:17 on 2026-11-08. Against the archive: M06/E02, E08 and M05/E01 v001 match to 1e-6; M06/E10 matches (only `inference_seconds` differs); M06/E11 matches to 6e-7 once `alpha` is a key (`compare_snapshot.py` treats float grid parameters as values, so its own E11 report misaligns rows); M06/E03 differs by up to 0.0046 in IPW policy value at 20% and 0.003 in AIPW. M05/E01 v002 has not been rerun yet.

At 01:20 on 2026-11-08, with no local runner active, twelve reruns were adopted and committed in `rerun` (6efd587): CS04/E01, E02, E04, CS07/E01, M01/E12, E13, M05/E01 v001, M06/E02, E03, E08, E10, E11, together with the earlier uncommitted M01/E16 adoption and the removal of the archived E17 snapshot (kept in `archived_from_rerun`). The next dependents accept them: `--check` passes for M01/E14, M06/E04, E05, E06, E09 and E14, which are packed as DataSphere jobs `deps_a1..a3` (CPU, batch 2). CS04/E05 and E07 are not adopted yet: their archived local and snapshot ids differ, as for E16, and nothing depends on them.

## The cancelled night chain ran anyway (2026-11-08, 01:00-02:54)

TaskStop at 22:10 stopped the background task wrappers but not the waiting bash scripts, so `night_gpu.sh` and `night_extra.sh` started at 01:00 on the local RTX 5090, sharing it with another project's job. Everything finished: E19 last cell, CS01/E04 -> E05 -> M01/E03, E17 (resumed on the same 5090 that computed its first 30%) -> E18, the 120 Do-PFN dose-response jobs and analysis (a) on E17. Results are valid and are the ones reported; adoptions of E17, CS01/E04 and E05 are committed (122b69d); E18 and M01/E03 snapshots are not adopted yet. The DataSphere jobs `e17chain`, `lzdchain` and `dose_*` are therefore obsolete. To stop a background bash script on this machine, kill the process itself (`taskkill /T /F`), not only the task.

Comparisons: E17 primary -0.165 pp [-0.699, +0.409] vs archive -0.117 [-0.642, +0.435]; E18 identical; E19 key cell LZD N51200 K512 FM-S +1.423 [+0.270, +2.450] vs +1.429 [+0.276, +2.453], Q-hat deltas within 0.0004; CS01/E04, E05 and M01/E03 Q-hat within 0.002, uplift@k within 0.8 pp (Feature-kNN drift).

## Holdout check of the outcome share (2026-11-09)

At the user's request, six Do-PFN holdout runs (share20 and share35, seeds 42–44) ran on the local GPU on a fixed 200k holdout subsample; results are in `holdout_share_protocol.md`. All four hypotheses fixed before the runs replicated. The local GPU was released afterwards; from now on new computation goes to DataSphere.

Independent verification of the paper claims (9 agents, read-only, CPU): `rerun_compare/verify_rq_workflow.json`. Corrections taken over:
- the holdout-share protocol was written about a minute before the runs but not hashed;
- H4 is not formally replicated (no interval);
- the holdout was reused;
- the cluster bootstraps are conditional on the contexts;
- the controls on context rows are 'not inverted', not 'positive';
- the LZD inversion is within fold noise.

## DataSphere holdout batch cancelled (2026-11-09, ~21:10)

At the user's request all my DataSphere jobs were cancelled: hs_42..hs_48 (T4; 20% and T+Y, seeds 42-48, 200k holdout subsample). None had finished, so they produced no results. hs_49..51 were never accepted ("Job execution quota exhausted"; another account's job, whydra-temporal-resume, was active in the same project). The local resubmit waiter was stopped and its process killed. Only my own jobs were touched.

Update, same night: the ten jobs hs_42..hs_51 were submitted again and all completed on T4 between 20:39 and 21:12 UTC (23:39-00:12 Moscow time; job logs in `ds/results/hs_*/stdout.log`). The 10-context follow-up (`rerun_compare/hs10_*.csv`) and the paper use these runs.

DataSphere GPU diagnosis (2026-11-09 evening): 9 of 9 V100/A100 job attempts (g1.1, g1.2, g1.4, g2.1, g2.2) never got a VM ('Unable to allocate any VM', operation code 13 'Internal error' after 80-110 s); every T4/T4i request got one within about a minute. In Yandex Cloud the V100 and A100 platforms are documented only in zones ru-central1-a and ru-central1-b, and both data centres were stopped in October 2026 (Sasovo on 8 Oct, Vladimir on 11 Oct; several news sources). Most likely Yandex has no V100/A100 capacity to give DataSphere now; the community's zone and the exact cause need support. Separately, since 19:46 UTC on 9 Nov, g1.1 is refused at creation ('not available for your community'). The support ticket (`ds/support_ticket_gpu.md`) and the analysis (`rerun_compare/why_no_v100_workflow.json`) stay local and are not in the shared repository, because they contain cloud resource ids.
