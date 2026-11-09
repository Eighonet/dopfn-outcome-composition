#!/usr/bin/env python3
"""Create a manifest of CV jobs for the Slurm runner."""
import argparse
import itertools
import json
from pathlib import Path

DATASETS = ("hillstrom", "retailhero", "orange", "criteo", "lzd")
MODELS = ("causalpfn", "tabpfn_t", "dopfn")
STRATEGIES = ("random", "balanced_treatment", "balanced_treatment_outcome")


def csv_items(s, allowed=None, numeric=False):
    values = [p.strip() for p in s.split(",")]
    if not values or any(not x for x in values):
        raise argparse.ArgumentTypeError(f"Invalid comma-separated list: {s!r}")
    if numeric:
        try:
            return [int(x) for x in values]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    if allowed is not None and any(x not in allowed for x in values):
        raise argparse.ArgumentTypeError(f"Unknown item in {s!r}; allowed: {', '.join(allowed)}")
    return values


def main():
    p = argparse.ArgumentParser(description="Create manifest; never submits Slurm jobs")
    p.add_argument("--datasets", default="hillstrom")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--folds", default="0")
    p.add_argument("--seeds", default="42")
    p.add_argument("--contexts", default="512")
    p.add_argument("--strategies", default="balanced_treatment")
    p.add_argument("--eval-max-size", type=int, default=8000)
    p.add_argument("--results-root", default="~/projects/smiles/results/fm_cv")
    p.add_argument("--manifest", required=True)
    p.add_argument("--skip-completed", action="store_true")
    a = p.parse_args()
    datasets = csv_items(a.datasets, DATASETS)
    models = csv_items(a.models, MODELS)
    folds = csv_items(a.folds, numeric=True)
    seeds = csv_items(a.seeds, numeric=True)
    contexts = csv_items(a.contexts, numeric=True)
    strategies = csv_items(a.strategies, STRATEGIES)
    if any(f < 0 or f >= 5 for f in folds):
        p.error("Folds must be between 0 and 4")
    if any(s not in (42, 43, 44) for s in seeds):
        p.error("Current fm_benchmark supports model seeds 42,43,44")
    if any(s not in (512,1024,2048,4096) for s in contexts):
        p.error("Current fm_benchmark supports context sizes 512,1024,2048,4096")
    if a.eval_max_size < 40:
        p.error("Eval max size must be >=40")
    manifest = Path(a.manifest).expanduser()
    if manifest.exists():
        p.error(f"Manifest already exists: {manifest}. Use a new name to preserve job-ID mapping")
    results_root = Path(a.results_root).expanduser()
    jobs, skipped = [], 0
    for dataset, model, fold, seed, context, strategy in itertools.product(
            datasets, models, folds, seeds, contexts, strategies):
        stem = (f"fold{fold}_{model}_{strategy}_ctx{context}_seed{seed}"
                f"_eval{a.eval_max_size}_cuda0")
        result_path = results_root / dataset / (stem + ".json")
        pred_path = results_root / dataset / (stem + ".predictions.npz")
        if a.skip_completed and result_path.is_file() and pred_path.is_file():
            try:
                if json.loads(result_path.read_text(encoding="utf-8")).get("status") == "ok":
                    skipped += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        jobs.append(f"{dataset} {model} {fold} {seed} {context} {strategy} {a.eval_max_size}\n")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as f:
        f.writelines(jobs)
    print(f"Manifest: {manifest}")
    print(f"Jobs ready: {len(jobs)}; previously complete: {skipped}")
    if jobs:
        print(f"Array range: 0-{len(jobs)-1}  (recommend concurrency %1)")
        print("First:", jobs[0].strip())
        print("Last: ", jobs[-1].strip())
    else:
        print("All selected experiments already complete; no array needed")


if __name__ == "__main__":
    main()
