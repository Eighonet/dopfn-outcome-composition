#!/usr/bin/env python3
"""Test the dense cache with synthetic data."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from fm_dense_cache import preprocess_cached_dense, cached_dense_load
from gpu_smoke import preprocess_train_only


def main():
    rng = np.random.default_rng(123)
    columns = ['f0', 'f1', 'f2', 'f3']
    values = rng.normal(size=(1200, len(columns))).astype(np.float64)
    values[6, 0] = np.nan
    values[12, 1] = np.inf
    values[18, 2] = -np.inf
    values[24, 3] = np.nan
    original = pd.DataFrame(values, columns=columns)
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / 'data.npy'
        np.save(p, values)
        raw = np.load(p, mmap_mode='r')
        for i, context_size in enumerate((32, 128, 256)):
            ctx = rng.choice(len(values), context_size, replace=False)
            ev = rng.choice(np.setdiff1d(np.arange(len(values)), ctx), 400, replace=False)
            oc, oe = preprocess_train_only(original, ctx, ev)
            nc, ne = preprocess_cached_dense(raw, ctx, ev, columns)
            np.testing.assert_array_equal(nc, oc)
            np.testing.assert_array_equal(ne, oe)
            print(f'PASS: preprocessing equality ctx={context_size}, eval=400')
    print('FINAL: PASS')


if __name__ == '__main__':
    main()
