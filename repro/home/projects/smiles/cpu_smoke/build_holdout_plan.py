#!/usr/bin/env python3
"""Create the fixed holdout plan from the selected CV configurations."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import pandas as pd

DATASETS = ("hillstrom", "retailhero", "orange", "criteo", "lzd")
MODELS = ("causalpfn", "tabpfn_t", "dopfn")
STRATEGIES = ("random", "balanced_treatment", "balanced_treatment_outcome")
SEEDS = (42, 43, 44)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="~/projects/smiles/results/analysis/fm_best_configurations.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--eval-max-size", type=int, default=8000)
    ap.add_argument("--full-holdout", action="store_true")
    a = ap.parse_args()
    target = Path(a.output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=False)
    source = Path(a.summary).expanduser().resolve()
    best = pd.read_csv(source)
    if len(best) != 15 or best.duplicated(["dataset", "model"]).any():
        raise ValueError("Expected exactly one CV-selected configuration for every dataset × model")
    assert set(best["dataset"]) == set(DATASETS)
    assert set(best["model"]) == set(MODELS)
    assert set(best["strategy"]).issubset(STRATEGIES)
    assert set(best["context"]).issubset((512, 1024, 2048, 4096))
    control = pd.DataFrame([{"dataset":"criteo", "model":"dopfn", "context":1024,
                             "strategy":"balanced_treatment"}])
    plan = pd.concat([best[["dataset","model","context","strategy"]], control],
                     ignore_index=True).drop_duplicates()
    assert len(plan) == 16, "The predefined control must add exactly one new configuration"
    rows = []
    for _, r in plan.iterrows():
        for seed in SEEDS:
            rows.append((r.dataset, r.model, int(r.context), r.strategy, int(seed)))
    manifest = target / "frozen_holdout_manifest.txt"
    manifest.write_text("".join(" ".join(map(str, row)) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    meta = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_selected_configurations":15, "n_extra_controls":1,
        "n_runs":len(rows), "seeds":list(SEEDS),
        "primary_metric":"qini_auc_normalized",
        "primary_controlled_comparison":"criteo dopfn ctx1024 balanced_treatment_outcome vs balanced_treatment",
        "eval_mode":"full_holdout" if a.full_holdout else "fixed_stratified_subset",
        "eval_max_size":None if a.full_holdout else a.eval_max_size,
        "manifest_sha256":digest,
        "source_best_csv":str(source),
        "source_best_csv_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (target / "frozen_holdout_meta.json").write_text(json.dumps(meta,indent=2), encoding="utf-8")
    print(f"FROZEN: {len(plan)} configs, {len(rows)} jobs")
    print("Manifest:", manifest)
    print("SHA256:", digest)
    print("Mode:", meta["eval_mode"])
    print("Do not modify after this point")

if __name__ == "__main__":
    main()
