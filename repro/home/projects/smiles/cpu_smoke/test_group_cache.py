#!/usr/bin/env python3
"""Test the group split cache with synthetic data."""
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
from fm_group_cache import cached_split


def sha(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "data"
    cache_root = Path(tmp) / "cache"
    cache_root.mkdir()
    base = root / "LZD"
    base.mkdir(parents=True)
    for name in ("features.parquet", "outcomes.parquet"):
        (base / name).write_bytes(name.encode())
    labels = np.array([-1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int8)
    np.save(cache_root / "lzd_fold_membership.npy", labels)
    folds = []
    for fold in range(5):
        tr = np.flatnonzero((labels >= 0) & (labels != fold))
        ev = np.flatnonzero(labels == fold)
        folds.append({"fold": fold, "n_train": len(tr), "n_eval_full": len(ev),
                      "train_sha256": sha(tr), "eval_sha256": sha(ev)})
    meta = {"dataset": "lzd", "n_rows": 12, "n_features": 1,
            "features": ["f0"],
            "source_file_stats": {
                name: {"size_bytes": (base / name).stat().st_size,
                       "mtime_ns": (base / name).stat().st_mtime_ns}
                for name in ("features.parquet", "outcomes.parquet")},
            "grouping": "pandas.util.hash_pandas_object(X[features], index=False), uint64",
            "final_fraction_requested": .15, "final_split_seed": 42,
            "outer_folds": 5, "outer_seed": 42,
            "n_final_holdout": 2, "final_holdout_hash": sha(np.flatnonzero(labels == -1)),
            "folds": folds, "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest()}
    (cache_root / "lzd_fold_membership.json").write_text(json.dumps(meta))
    for fold in range(5):
        tr, ev, final, method, tag = cached_split(root, "lzd", 12, ["f0"], fold, cache_root)
        assert len(tr) == 8 and len(ev) == 2 and len(final) == 2
        assert not np.intersect1d(tr, ev).size
        assert not np.intersect1d(tr, final).size
        assert "cached" in method and tag == meta["labels_sha256"]
    print("PASS: all 5 cached folds")
    try:
        cached_split(root, "lzd", 13, ["f0"], 0, cache_root)
        raise AssertionError("mismatched n_rows accepted")
    except ValueError:
        print("PASS: refuse different row count")
    (base / "features.parquet").write_bytes(b"changed")
    try:
        cached_split(root, "lzd", 12, ["f0"], 0, cache_root)
        raise AssertionError("modified Parquet accepted")
    except ValueError:
        print("PASS: refuse changed Parquet source")
print("FINAL: PASS")
