"""Parallel twin of bootstrap_criteo_dopfn.py: the same checks, the same draws, the same output columns.

The resampling draws come from one generator in the main process, in the original order, so every
replicate is identical to the sequential script; only the metric computation is spread over workers.
Usage: bootstrap_parallel.py --reps 1000 --workers 12
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(os.environ["CODE"])
sys.path.insert(0, str(CODE))
import bootstrap_criteo_dopfn as original  # noqa: E402  the archived script, unmodified

STATE = {}


def init(scores, y, t, cluster_id):
    STATE.update(scores=scores, y=y, t=t, cluster_id=cluster_id,
                 row_ids=np.arange(len(y), dtype=np.int32))


def one(job):
    iteration, multiplicity = job
    ix = np.repeat(STATE["row_ids"], multiplicity[STATE["cluster_id"]])
    yy, tt = STATE["y"][ix], STATE["t"][ix]
    row = {"rep": iteration, "n_eval_boot": len(ix)}
    for seed in original.SEEDS:
        base = original.measures(STATE["scores"][seed, "balanced_treatment"][ix], yy, tt)
        alt = original.measures(STATE["scores"][seed, "balanced_treatment_outcome"][ix], yy, tt)
        for metric in ("qini", "auuc", "uplift10"):
            row[f"{metric}_delta_seed{seed}"] = alt[metric] - base[metric]
    for metric in ("qini", "auuc", "uplift10"):
        row[f"{metric}_delta_mean"] = np.mean([row[f"{metric}_delta_seed{seed}"] for seed in original.SEEDS])
    return row


def load():
    """Same inputs and checks as the original main(), up to the resampling loop."""
    import hashlib
    project = Path.home() / "projects/smiles"
    manifest = project / "manifests/holdout_frozen_20261021_233147/frozen_holdout_manifest.txt"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == original.EXPECTED_SHA, "Frozen plan changed"
    data_dir = project / "results/fm_holdout/criteo"
    scores, reference, y, t, eval_indices = {}, {}, None, None, None
    for seed in original.SEEDS:
        for strategy in original.STRATEGIES:
            stem = f"holdout_dopfn_{strategy}_ctx1024_seed{seed}_evalall_cuda0"
            d = json.loads((data_dir / f"{stem}.json").read_text())
            assert d["status"] == "ok" and d["eval_scope"] == "full_holdout"
            assert d["selection_manifest_sha256"] == original.EXPECTED_SHA
            with np.load(data_dir / f"{stem}.predictions.npz", allow_pickle=False) as z:
                pred = np.asarray(z["predictions"], dtype=np.float64)
                if y is None:
                    y, t = np.asarray(z["y"], dtype=np.int8), np.asarray(z["t"], dtype=np.int8)
                    eval_indices = np.asarray(z["eval_indices"], dtype=np.int64)
                else:
                    assert np.array_equal(y, z["y"]) and np.array_equal(t, z["t"])
                    assert np.array_equal(eval_indices, z["eval_indices"])
                assert np.isfinite(pred).all() and len(pred) == len(y) == d["n_eval"]
                scores[seed, strategy] = pred
                m = d["metrics"]
                reference[seed, strategy] = {"qini": float(m["qini_auc_normalized"]), "auuc": float(m["auuc"]),
                                             "uplift10": float(m["uplift@10%"])}
    for key, pred in scores.items():
        measured = original.measures(pred, y, t)
        for metric, value in measured.items():
            assert np.isclose(value, reference[key][metric], rtol=1e-8, atol=1e-10), (key, metric)
    print("All 18 original metric values reproduced: PASS", flush=True)
    full_x = np.load(project / "cache/dense_features/criteo_X_float64.npy", mmap_mode="r")
    assert full_x.shape[0] > int(eval_indices.max()) and full_x.shape[1] == 10
    hashes = pd.util.hash_pandas_object(pd.DataFrame(np.asarray(full_x[eval_indices])), index=False).to_numpy(dtype=np.uint64)
    _, cluster_id = np.unique(hashes, return_inverse=True)
    return scores, reference, y, t, cluster_id, project


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=20261027)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--part", type=int, default=0)
    p.add_argument("--parts", type=int, default=1)
    p.add_argument("--out", default="fm_criteo_dopfn_cluster_bootstrap_B{reps}_parallel.csv")
    a = p.parse_args()
    scores, reference, y, t, cluster_id, project = load()
    n_clusters = int(cluster_id.max()) + 1
    print(f"Feature-vector clusters: {n_clusters:,}", flush=True)
    rng = np.random.default_rng(a.bootstrap_seed)

    def draws():
        for iteration in range(1, a.reps + 1):
            selected = rng.integers(0, n_clusters, size=n_clusters, dtype=np.int32)
            yield iteration, np.bincount(selected, minlength=n_clusters)

    start, results = time.perf_counter(), []
    if a.parts > 1:
        # Windows cannot spawn workers from this venv, so independent processes each take every n-th
        # replicate; all of them walk the same generator in order, so the draws are unchanged.
        init(scores, y, t, cluster_id)
        for iteration, multiplicity in draws():
            if (iteration - 1) % a.parts == a.part:
                results.append(one((iteration, multiplicity)))
                print(f"part {a.part}: replicate {iteration}, elapsed {time.perf_counter() - start:.1f}s", flush=True)
        out = project / "results/analysis" / f"bootstrap_parallel_part{a.part}_of{a.parts}.csv"
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"PART DONE: {out}")
        return
    with Pool(a.workers, initializer=init, initargs=(scores, y, t, cluster_id)) as pool:
        for row in pool.imap(one, draws(), chunksize=1):
            results.append(row)
            if row["rep"] == 1 or row["rep"] % 5 == 0 or row["rep"] == a.reps:
                print(f"Bootstrap {row['rep']}/{a.reps}, elapsed {time.perf_counter() - start:.1f}s", flush=True)
    df = pd.DataFrame(results)
    out = project / "results/analysis" / a.out.format(reps=a.reps)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n===== PAIRED CLUSTER BOOTSTRAP =====")
    for metric in ("qini", "auuc", "uplift10"):
        obs = np.mean([reference[s, "balanced_treatment_outcome"][metric] - reference[s, "balanced_treatment"][metric]
                       for s in original.SEEDS])
        lo, hi = np.percentile(df[f"{metric}_delta_mean"].to_numpy(), [2.5, 97.5])
        print(f"{metric:9s} observed={obs:+.5f} bootstrap95=[{lo:+.5f}, {hi:+.5f}]")
    print(f"Results: {out}")
    print(f"Total bootstrap time: {time.perf_counter() - start:.1f}s")
    print("FINAL: PASS")


if __name__ == "__main__":
    main()
