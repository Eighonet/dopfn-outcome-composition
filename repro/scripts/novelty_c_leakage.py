"""(c) Does a naive row split inflate uplift metrics on Criteo through duplicated feature vectors?

Same 2M-row uniform sample, two 80/20 splits: naive (stratified by T x Y) and group-safe (identical
feature vectors kept on one side, as in the report). Same-size train bank (400k) and eval (100k) in both.
Models: S-Learner (HistGB, M01/E02 settings) and the plain neighbour mean (256 treated - 256 control
nearest train rows, standardized features), i.e. the local estimator that matched the FM in HypeCheck.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklift.metrics import qini_auc_score

DATA = Path(sys.argv[1]) / "Criteo-ITE-v2.1"
OUT = Path(sys.argv[2])
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42      # repeats vary the sample and both splits
SAMPLE, BANK, EVAL, K = 2_000_000, 400_000, 100_000, 512
start = time.perf_counter()
feats = pd.read_parquet(DATA / "features.parquet")
visit = pd.read_parquet(DATA / "outcomes.parquet", columns=["epk_id", "visit"])
assert (feats.epk_id.to_numpy() == visit.epk_id.to_numpy()).all()
rows = np.sort(np.random.default_rng(SEED).choice(len(feats), SAMPLE, replace=False))
cols = [c for c in feats.columns if c.startswith("f")]
x = feats[cols].to_numpy(np.float32)[rows]
t = feats["T"].to_numpy(np.int8)[rows]
y = visit.visit.to_numpy(np.int8)[rows]
groups = pd.util.hash_pandas_object(pd.DataFrame(x), index=False).to_numpy()
del feats, visit
print(f"loaded {SAMPLE:,} rows, {len(np.unique(groups)):,} distinct feature vectors, "
      f"{time.perf_counter() - start:.0f}s", flush=True)


def uplift_at(score, yy, tt, k=0.10):
    top = np.argsort(-score, kind="stable")[:int(len(score) * k)]
    return 100 * (yy[top][tt[top] == 1].mean() - yy[top][tt[top] == 0].mean())


strata = 2 * t + y
splits = {"naive": train_test_split(np.arange(SAMPLE), test_size=0.2, random_state=SEED, stratify=strata),
          "group_safe": next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(x, strata, groups))}
out = []
for name, (train, test) in splits.items():
    rng = np.random.default_rng(SEED + 1)
    bank, ev = np.sort(rng.choice(train, BANK, replace=False)), np.sort(rng.choice(test, EVAL, replace=False))
    seen = np.isin(groups[ev], groups[train])
    x_b, t_b, y_b, x_e, t_e, y_e = x[bank], t[bank], y[bank], x[ev], t[ev], y[ev]
    s = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=SEED)
    s.fit(np.column_stack([x_b, t_b]), y_b)
    s_score = (s.predict_proba(np.column_stack([x_e, np.ones(EVAL)]))[:, 1] -
               s.predict_proba(np.column_stack([x_e, np.zeros(EVAL)]))[:, 1])
    mu, sd = x_b.mean(0), x_b.std(0) + 1e-6
    local = np.zeros(EVAL)
    for arm, sign in ((1, 1), (0, -1)):
        idx = np.flatnonzero(t_b == arm)
        nn = NearestNeighbors(n_neighbors=K // 2, n_jobs=int(os.environ.get("KNN_JOBS", "4"))).fit((x_b[idx] - mu) / sd)
        neigh = nn.kneighbors((x_e - mu) / sd, return_distance=False)
        local += sign * y_b[idx][neigh].mean(1)
    for model, score in (("s_learner", s_score), ("neighbour_mean", local)):
        # 'all' is the headline; 'unseen' asks whether the naive gap survives without duplicated vectors
        for subset, mask in (("all", np.ones(EVAL, bool)), ("seen", seen), ("unseen", ~seen)):
            if mask.sum() < 1000:
                continue
            row = dict(seed=SEED, split=name, model=model, subset=subset, rows=int(mask.sum()),
                       eval_seen_share=float(seen.mean()),
                       qhat=float(qini_auc_score(y_e[mask], score[mask], t_e[mask])),
                       uplift10_pp=float(uplift_at(score[mask], y_e[mask], t_e[mask])))
            out.append(row)
            print(row, f"{time.perf_counter() - start:.0f}s", flush=True)
pd.DataFrame(out).to_csv(OUT, index=False)
