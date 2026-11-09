"""(b) Outcome-share dose-response on the report's CV folds, with the unmodified fm_benchmark.

Adds strategies shareNN: treatment-balanced context (size/2 per arm) in which NN% of each arm has
Y=1. balanced_treatment (natural share) and balanced_treatment_outcome (50%) already exist in the CV
results, so together they give the curve. Seeding follows choose_context: default_rng(seed + fold).
Usage like run_cv.py: run_dose_cv.py <manifest> with lines "dataset model fold seed K strategy eval".
"""
import os
import sys
import time

import numpy as np

manifest = os.path.abspath(sys.argv[1])
code = os.environ["CODE"]; os.chdir(code); sys.path.insert(0, code)
import fm_benchmark as fm

SHARES = (10, 20, 35, 75)
fm.STRATEGIES = fm.STRATEGIES + tuple(f"share{p}" for p in SHARES)
original = fm.choose_context


def choose_context(tr, t, y, size, strategy, fold, context_seed=fm.CONTEXT_SEED):
    if not strategy.startswith("share"):
        return original(tr, t, y, size, strategy, fold, context_seed)
    share = int(strategy[5:]) / 100
    rng = np.random.default_rng(context_seed + fold)
    half = size // 2
    ids = []
    for arm, n_arm in ((0, size - half), (1, half)):
        n_pos = int(round(n_arm * share))
        for value, n in ((0, n_arm - n_pos), (1, n_pos)):
            cell = tr[(t[tr] == arm) & (y[tr] == value)]
            if len(cell) < n:
                raise ValueError(f"{strategy}: arm {arm} y={value} has {len(cell)} rows, needs {n}")
            ids.append(rng.choice(cell, n, replace=False))
    return rng.permutation(np.concatenate(ids))


fm.choose_context = choose_context
jobs = [line.split() for line in open(manifest) if line.strip()]
extra = sys.argv[2:]   # e.g. --preview --device cpu for a dry check
t0 = time.time()
for i, (d, m, f, s, k, st, ev) in enumerate(jobs, 1):
    sys.argv = ["fm_benchmark.py", "--dataset", d, "--model", m, "--fold", f, "--seed", s, "--context-size", k,
                "--strategy", st, "--eval-max-size", ev, "--output", os.environ["FM_OUTPUT"], *extra]
    t = time.time(); rc = fm.main()
    print(f"### job {i}/{len(jobs)} rc={rc} {time.time() - t:.1f}s elapsed={time.time() - t0:.0f}s "
          f"{d} {m} f{f} s{s} K{k} {st}", flush=True)
