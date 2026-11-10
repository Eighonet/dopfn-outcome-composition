#!/usr/bin/env python3
"""Build a raw numeric feature cache for Criteo without fitting preprocessing."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

import fm_benchmark as fm
from fm_dense_cache import (
    CACHE_DIR, FORMAT, SPLIT_DIR, paths, source_stats, sha256_file,
    preprocess_cached_dense, cached_dense_load,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='~/projects/smiles/data/data_A_cleaned')
    p.add_argument('--cache-dir', default=str(CACHE_DIR))
    a = p.parse_args()
    start = time.monotonic()
    root = Path(a.data_root).expanduser().resolve()
    cache_dir = Path(a.cache_dir).expanduser().resolve()
    xp, tp, yp, jp = paths(cache_dir)
    if any(path.exists() for path in (xp, tp, yp, jp)):
        raise FileExistsError('Dense cache already exists or partially created. Stop and inspect; never overwrite automatically.')
    cache_dir.mkdir(parents=True, exist_ok=True)
    sp = SPLIT_DIR / 'criteo_fold_membership.json'
    split = json.loads(sp.read_text(encoding='utf-8'))
    if split['source_file_stats'] != source_stats(root):
        raise RuntimeError('Source Parquet changed since grouped-split cache was created')

    print('Loading Criteo once with safe ID alignment...', flush=True)
    X, t, y, _, features = fm.load(root, 'criteo', use_group_hashes=False)
    n, f = len(t), len(features)
    if (n != split['n_rows'] or features != split['features']):
        raise ValueError('Parquet data and validated grouped-split cache differ')
    bad = [c for c in features if not (is_numeric_dtype(X[c]) or is_bool_dtype(X[c]))]
    if bad:
        raise TypeError(f'Numeric dense cache is only valid for numeric columns. Found: {bad}')
    print(f'Writing raw float64 X: {n:,} x {f} = {n*f*8/1024**3:.2f} GiB', flush=True)
    tmpx, tmpt, tmpy = [path.with_name(path.name + '.incomplete.npy') for path in (xp, tp, yp)]
    for pth in (tmpx, tmpt, tmpy):
        if pth.exists():
            raise FileExistsError(f'Old incomplete cache: {pth}. Inspect before removing.')
    arr = np.lib.format.open_memmap(tmpx, mode='w+', dtype=np.float64, shape=(n, f))
    for j, feature in enumerate(features):
        arr[:, j] = pd.to_numeric(X[feature], errors='raise').to_numpy(dtype=np.float64, na_value=np.nan)
        print(f'  column {j+1}/{f}: {feature}', flush=True)
    arr.flush()
    del arr
    np.save(tmpt, np.asarray(t, dtype=np.int8), allow_pickle=False)
    np.save(tmpy, np.asarray(y, dtype=np.int8), allow_pickle=False)

    print('Verifying cached preprocessing against original preprocessing...', flush=True)
    # Select a fixed row sample without using treatment or outcome.
    rng = np.random.default_rng(7717)
    ctx = rng.choice(n, size=min(256, n//3), replace=False)
    ev = rng.choice(np.setdiff1d(np.arange(n), ctx), size=min(512, n//3), replace=False)
    from gpu_smoke import preprocess_train_only
    expected_ctx, expected_ev = preprocess_train_only(X, ctx, ev)
    raw = np.load(tmpx, mmap_mode='r', allow_pickle=False)
    actual_ctx, actual_ev = preprocess_cached_dense(raw, ctx, ev, features)
    if not (np.array_equal(expected_ctx, actual_ctx, equal_nan=True) and
            np.array_equal(expected_ev, actual_ev, equal_nan=True)):
        raise AssertionError('Cached preprocess differs from original; leave old benchmark unchanged')
    print('PREPROCESS REFERENCE MATCH: PASS', flush=True)
    del X, raw

    os.replace(tmpx, xp)
    os.replace(tmpt, tp)
    os.replace(tmpy, yp)
    info = {
        'format': FORMAT, 'dataset': 'criteo', 'n_rows': n,
        'feature_columns': list(features),
        'dtypes': {'X': 'float64', 't': 'int8', 'y': 'int8'},
        'source_file_stats': source_stats(root),
        'split_labels_sha256': split['labels_sha256'],
        'file_sha256': {path.name: sha256_file(path) for path in (xp, tp, yp)},
        'preprocess_reference_match': True,
    }
    tmpmeta = jp.with_name(jp.name + '.tmp')
    tmpmeta.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmpmeta, jp)
    # Check that the saved cache can be loaded.
    _, t2, y2, _, _ = cached_dense_load(root, cache_dir)
    if not (np.array_equal(t2, t) and np.array_equal(y2, y)):
        raise AssertionError('T/Y cache mismatch')
    print(f'PASS: saved {xp} ({xp.stat().st_size/1024**3:.2f} GiB)', flush=True)
    print(f'METADATA: {jp}', flush=True)
    print(f'Time: {time.monotonic()-start:.1f}s', flush=True)


if __name__ == '__main__':
    sys.exit(main())
