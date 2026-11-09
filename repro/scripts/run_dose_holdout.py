"""Holdout runs with an outcome-share context (protocol: repro/holdout_share_protocol.md).

Runs the unmodified fm_holdout.py with the shareNN strategies of scripts/run_dose_cv.py added:
a treatment-balanced context in which NN% of each arm has Y=1, seeded like choose_context.
Usage: run_dose_holdout.py <strategy> <seed> [extra fm_holdout args, e.g. --preview --device cpu]
Needs HOME/USERPROFILE = repro/home and CODE = the cpu_smoke folder, as for the original holdout runs.
"""
import os
import sys

import numpy as np

code = os.environ["CODE"]; os.chdir(code); sys.path.insert(0, code)
import fm_benchmark as fm
import fm_holdout

SHARES = (10, 20, 35, 75)
fm.STRATEGIES = fm.STRATEGIES + tuple(f"share{p}" for p in SHARES)
# more context draws for the paired 20% vs T+Y check (the seed only sets the context for Do-PFN)
fm.MODEL_SEEDS = tuple(sorted(set(fm.MODEL_SEEDS) | set(range(42, 52))))
original = fm.choose_context


def choose_context(tr, t, y, size, strategy, fold, context_seed=fm.CONTEXT_SEED):
    """Same sampler as scripts/run_dose_cv.py."""
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
strategy, seed, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.argv = ["fm_holdout.py", "--dataset", "criteo", "--model", "dopfn", "--seed", seed, "--context-size", "1024",
            "--strategy", strategy, "--eval-max-size", os.environ.get("EVAL_MAX", "200000"),
            "--output", os.environ["HOLDOUT_OUT"], *extra]
sys.exit(fm_holdout.main())
