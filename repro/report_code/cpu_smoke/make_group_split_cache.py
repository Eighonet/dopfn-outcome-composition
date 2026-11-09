#!/usr/bin/env python3
"""Build deterministic group-safe split indices for Criteo and LZD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def hash_indices(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr, dtype=np.int64).tobytes()).hexdigest()


def source_signature(root: Path, name: str, fm) -> dict:
    base = root / fm.DATASETS[name]['folder']
    files = {}
    for fname in ('features.parquet','outcomes.parquet'):
        p = base / fname
        st = p.stat()
        files[fname] = {'size_bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
    return files


def compute_labels(t: np.ndarray, y: np.ndarray, groups: np.ndarray, fm):
    n = len(t)
    assert len(y) == n and len(groups) == n
    ids = np.arange(n, dtype=np.int64)
    strata = t.astype(np.int8) * 2 + y.astype(np.int8)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=fm.FINAL_TEST_SIZE,
        random_state=fm.FINAL_TEST_SEED,
    )
    dev, hold = next(splitter.split(ids, strata, groups))
    outer = StratifiedGroupKFold(
        n_splits=fm.OUTER_FOLDS, shuffle=True,
        random_state=fm.OUTER_SPLIT_SEED,
    )
    labels = np.full(n, -2, dtype=np.int8)
    labels[hold] = -1
    for fold, (_, val_ix) in enumerate(
        outer.split(ids[dev], strata[dev], groups[dev])
    ):
        ev = dev[val_ix]
        if np.any(labels[ev] != -2):
            raise AssertionError('Outer evaluation folds overlap')
        labels[ev] = fold
    if np.any(labels == -2):
        raise AssertionError('Some rows have no partition assignment')
    # Keep identical feature vectors in the same split.
    # Sort group labels without looping over individual rows.
    ix = np.argsort(groups, kind='stable')
    sorted_g = groups[ix]
    sorted_labels = labels[ix]
    if np.any((sorted_g[1:] == sorted_g[:-1]) &
              (sorted_labels[1:] != sorted_labels[:-1])):
        raise AssertionError('Duplicate feature-vector group crossed split boundary')
    return labels


def validate_against_reference(labels, t, y, groups, fm, folds=(0,)):
    # Compare split assignments with the existing reference function.
    for fold in folds:
        train, ev, holdout, method = fm.split_cv(t, y, groups, fold)
        got_train = np.flatnonzero((labels >= 0) & (labels != fold))
        got_ev = np.flatnonzero(labels == fold)
        got_hold = np.flatnonzero(labels == -1)
        if not (np.array_equal(train, got_train) and
                np.array_equal(ev, got_ev) and
                np.array_equal(holdout, got_hold)):
            raise AssertionError(
                f'Fold {fold}: cache indices/order differ from existing split_cv'
            )
        print(f'REFERENCE MATCH fold={fold} method={method} '
              f'train={len(train):,} eval={len(ev):,} holdout={len(holdout):,}',
              flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=('lzd', 'criteo'))
    p.add_argument('--data-root', default='~/projects/smiles/data/data_A_cleaned')
    p.add_argument('--cache-root', default='~/projects/smiles/cache/group_splits')
    p.add_argument('--check-reference', action='store_true',
                   help='Extra costly check versus old split_cv for fold 0')
    a = p.parse_args()
    import fm_benchmark as fm
    if not (fm.DATASETS[a.dataset]['group'] and fm.OUTER_FOLDS == 5):
        raise RuntimeError('Unexpected research split settings')
    root = Path(a.data_root).expanduser().resolve()
    cache_root = Path(a.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f'{a.dataset}_fold_membership.npy'
    meta_path = cache_root / f'{a.dataset}_fold_membership.json'
    if path.exists() or meta_path.exists():
        raise FileExistsError('Cache exists; do not overwrite: ' + str(path))
    t0 = time.monotonic()
    print(f'Loading {a.dataset} once (this may take minutes)...', flush=True)
    X, t, y, groups, feature_names = fm.load(root, a.dataset)
    if groups is None:
        raise ValueError('No feature-vector groups; verify fm_benchmark.load()')
    del X
    print(f'Rows={len(t):,}; features={len(feature_names)}', flush=True)
    print('Computing 15% holdout and five grouped folds...', flush=True)
    labels = compute_labels(t, y, groups, fm)
    print('Group-leakage check: PASS', flush=True)
    if a.check_reference:
        print('Comparing fold 0 to original split_cv...', flush=True)
        validate_against_reference(labels, t, y, groups, fm)
    summary = []
    final = np.flatnonzero(labels == -1)
    for fold in range(fm.OUTER_FOLDS):
        tr = np.flatnonzero((labels >= 0) & (labels != fold))
        ev = np.flatnonzero(labels == fold)
        summary.append({'fold': fold, 'n_train': len(tr), 'n_eval_full': len(ev),
                        'train_sha256': hash_indices(tr), 'eval_sha256': hash_indices(ev)})
        print(f'fold {fold}: train={len(tr):,} eval={len(ev):,}', flush=True)
    st = source_signature(root, a.dataset, fm)
    meta = {
        'dataset': a.dataset, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'n_rows': len(t), 'n_features': len(feature_names),
        'features': feature_names,
        'source_file_stats': st,
        'grouping': 'pandas.util.hash_pandas_object(X[features], index=False), uint64',
        'group_hash_sha256': hashlib.sha256(np.ascontiguousarray(groups).tobytes()).hexdigest(),
        'n_groups': int(len(np.unique(groups))),
        'final_fraction_requested': fm.FINAL_TEST_SIZE,
        'final_split_seed': fm.FINAL_TEST_SEED,
        'outer_folds': fm.OUTER_FOLDS,
        'outer_seed': fm.OUTER_SPLIT_SEED,
        'n_final_holdout': len(final),
        'final_holdout_hash': hash_indices(final),
        'folds': summary,
        'labels_encoding': '-1 final holdout, 0..4 CV eval fold, every other non-final fold is training',
        'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
        'script_version': 1,
    }
    tmp = cache_root / f'.{path.name}.{os.getpid()}.tmp.npy'
    np.save(tmp, labels, allow_pickle=False)
    os.replace(tmp, path)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    print(f'PASS: saved {path} ({path.stat().st_size / (1024**2):.1f} MiB)', flush=True)
    print(f'METADATA: {meta_path}', flush=True)
    print(f'Time: {time.monotonic() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
