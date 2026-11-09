"""Measure E17 speed on this CPU under the current load: feature retrieval on spread eval rows, then the
runner's own native benchmark (4096 eval rows, 125.4 s on the original CPU). Run from repro/rerun."""
import os
import sys
import time

os.environ["REPRO_DEVICE"] = "cpu"
os.environ.setdefault("REPRO_BATCH", "2")
sys.path.insert(0, "tools")
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

import run_m01_e17_v001 as r

N = int(os.environ.get("CPU_SPEED_QUERIES", "256"))
batch = int(os.environ["REPRO_BATCH"])
start = time.perf_counter()
plan, train_ids, candidate, weights = r.verify_source()
data, x, cols, train_pos, eval_pos, eval_ids, admissible = r.build_eval(train_ids, candidate)
estimator = r.e15.e08.earlier.model_from_weights(weights)
a, q, _ = r.e15.context.preprocess(x, train_pos, eval_pos, estimator.max_feature_size)
tt = data["T"].to_numpy(dtype=np.int8)[train_pos]
yt = data.visit.to_numpy(dtype=np.int8)[train_pos]
print(f"setup {time.perf_counter() - start:.0f}s, torch threads {torch.get_num_threads()}, batch {batch}", flush=True)

arms = [np.flatnonzero(tt == arm) for arm in (0, 1)]
neighbours = [NearestNeighbors(n_neighbors=r.K // 2, algorithm="auto", n_jobs=1).fit(a[arm]) for arm in arms]
forward = r.e15.source.make_forward(estimator, torch.device("cpu"), a, tt, yt)
chosen = q[np.linspace(0, len(q) - 1, N, dtype=np.int64)]
started = time.perf_counter()
indices = np.concatenate([arm[nn.kneighbors(chosen, return_distance=False)]
                          for arm, nn in zip(arms, neighbours)], axis=1).astype(np.int32)
scores = np.concatenate([forward(indices[j:j + batch], chosen[j:j + batch]) for j in range(0, N, batch)])
seconds = time.perf_counter() - started
assert len(scores) == N and np.isfinite(scores).all()
print(f"FEATURE: {N} queries in {seconds:.1f}s = {N / seconds:.2f} query/s; "
      f"remaining 906,784 would take {906_784 / (N / seconds) / 3600:.1f}h", flush=True)
del estimator
native = r.e16.prepare_estimator(r.e15.e08.earlier.model_from_weights(weights), a, tt, yt)
r.benchmark_native(native, a, q, tt, yt)
