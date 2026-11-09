#!/usr/bin/env python3
"""Run a paired cluster bootstrap for the fixed Criteo Do-PFN comparison."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k

STRATEGIES = ('balanced_treatment', 'balanced_treatment_outcome')
SEEDS = (42, 43, 44)
EXPECTED_SHA = 'dd8d349f336399e438caf7be4f2a250746b137b1dbba9c5d79cb7292a0f884e2'


def measures(scores, y, t):
    return {
        'qini': float(qini_auc_score(y_true=y, uplift=scores, treatment=t)),
        'auuc': float(uplift_auc_score(y_true=y, uplift=scores, treatment=t)),
        'uplift10': float(uplift_at_k(y_true=y, uplift=scores, treatment=t,
                                    strategy='overall', k=0.1)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--bootstrap-seed', type=int, default=20261027)
    a = p.parse_args()
    assert a.reps > 0
    project = Path.home() / 'projects/smiles'
    manifest = project / 'manifests/holdout_frozen_20261021_233147/frozen_holdout_manifest.txt'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == EXPECTED_SHA, 'Frozen plan changed'
    data_dir = project / 'results/fm_holdout/criteo'
    output_dir = project / 'results/analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = {}
    reference = {}
    y = t = eval_indices = None
    for seed in SEEDS:
        for strategy in STRATEGIES:
            stem = f'holdout_dopfn_{strategy}_ctx1024_seed{seed}_evalall_cuda0'
            d = json.loads((data_dir / f'{stem}.json').read_text())
            assert d['status'] == 'ok' and d['eval_scope'] == 'full_holdout'
            assert d['selection_manifest_sha256'] == EXPECTED_SHA
            with np.load(data_dir / f'{stem}.predictions.npz', allow_pickle=False) as z:
                pred = np.asarray(z['predictions'], dtype=np.float64)
                if y is None:
                    y = np.asarray(z['y'], dtype=np.int8)
                    t = np.asarray(z['t'], dtype=np.int8)
                    eval_indices = np.asarray(z['eval_indices'], dtype=np.int64)
                else:
                    assert np.array_equal(y, z['y'])
                    assert np.array_equal(t, z['t'])
                    assert np.array_equal(eval_indices, z['eval_indices'])
                assert np.isfinite(pred).all()
                assert len(pred) == len(y) == d['n_eval']
                scores[seed, strategy] = pred
                m = d['metrics']
                reference[seed, strategy] = {
                    'qini': float(m['qini_auc_normalized']),
                    'auuc': float(m['auuc']),
                    'uplift10': float(m['uplift@10%']),
                }
    print(f'Loaded 6 prediction vectors, each of {len(y):,} test observations', flush=True)

    # Check the original metric values before resampling.
    for key, pred in scores.items():
        measured = measures(pred, y, t)
        for metric, value in measured.items():
            assert np.isclose(value, reference[key][metric], rtol=1e-8, atol=1e-10), (
                f'Metric reproduction failed: {key}, {metric}, '
                f'{value} != {reference[key][metric]}'
            )
    print('All 18 original metric values reproduced: PASS', flush=True)

    # Group test rows by their original feature vectors.
    feature_path = project / 'cache/dense_features/criteo_X_float64.npy'
    full_x = np.load(feature_path, mmap_mode='r')
    assert full_x.shape[0] > int(eval_indices.max()) and full_x.shape[1] == 10
    x_eval = np.asarray(full_x[eval_indices])
    hashes = pd.util.hash_pandas_object(pd.DataFrame(x_eval), index=False).to_numpy(dtype=np.uint64)
    del x_eval, full_x
    _, cluster_id = np.unique(hashes, return_inverse=True)
    del hashes
    n_clusters = int(cluster_id.max()) + 1
    csize = np.bincount(cluster_id, minlength=n_clusters)
    duplicated_rows = int(csize[csize > 1].sum())
    print(f'Feature-vector clusters: {n_clusters:,}; rows in repeated clusters: '
          f'{duplicated_rows:,} / {len(y):,}', flush=True)

    rng = np.random.default_rng(a.bootstrap_seed)
    results = []
    start = time.perf_counter()
    row_ids = np.arange(len(y), dtype=np.int32)
    for iteration in range(1, a.reps + 1):
        # Use the same resampled test rows for each strategy and seed.
        selected_groups = rng.integers(0, n_clusters, size=n_clusters, dtype=np.int32)
        multiplicity = np.bincount(selected_groups, minlength=n_clusters)
        ix = np.repeat(row_ids, multiplicity[cluster_id])
        yy, tt = y[ix], t[ix]
        row = {'rep': iteration, 'n_eval_boot': len(ix)}
        for seed in SEEDS:
            base = measures(scores[seed, 'balanced_treatment'][ix], yy, tt)
            alt = measures(scores[seed, 'balanced_treatment_outcome'][ix], yy, tt)
            for metric in ('qini', 'auuc', 'uplift10'):
                row[f'{metric}_delta_seed{seed}'] = alt[metric] - base[metric]
        for metric in ('qini', 'auuc', 'uplift10'):
            row[f'{metric}_delta_mean'] = np.mean([
                row[f'{metric}_delta_seed{seed}'] for seed in SEEDS
            ])
        results.append(row)
        if iteration == 1 or iteration % 5 == 0 or iteration == a.reps:
            print(f'Bootstrap {iteration}/{a.reps}, '
                  f'elapsed {time.perf_counter() - start:.1f}s', flush=True)

    df = pd.DataFrame(results)
    out = output_dir / f'fm_criteo_dopfn_cluster_bootstrap_B{a.reps}.csv'
    df.to_csv(out, index=False)
    print('\n===== PAIRED CLUSTER BOOTSTRAP =====')
    for metric in ('qini', 'auuc', 'uplift10'):
        obs = np.mean([
            reference[seed, 'balanced_treatment_outcome'][metric] -
            reference[seed, 'balanced_treatment'][metric]
            for seed in SEEDS
        ])
        sampled = df[f'{metric}_delta_mean'].to_numpy()
        lo, hi = np.percentile(sampled, [2.5, 97.5])
        print(f'{metric:9s} observed={obs:+.5f} '
              f'bootstrap95=[{lo:+.5f}, {hi:+.5f}]')
    print(f'Results: {out}')
    print(f'Total bootstrap time: {time.perf_counter()-start:.1f}s')
    print('FINAL: PASS')


if __name__ == '__main__':
    main()
