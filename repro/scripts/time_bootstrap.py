"""Time one bootstrap replicate at the real holdout size using saved Do-PFN predictions."""
import glob, os, time
import numpy as np
from sklift.metrics import qini_auc_score, uplift_auc_score, uplift_at_k
f = sorted(glob.glob(os.path.expanduser('~/projects/smiles/results/fm_holdout/criteo/holdout_dopfn_balanced_treatment_outcome_ctx1024_seed42_evalall_cuda0.predictions.npz')))[0]
z = np.load(f); p, y, t = z['predictions'].astype(float), z['y'], z['t']; n = len(y)
rng = np.random.default_rng(1); ix = rng.integers(0, n, n)
t0 = time.perf_counter(); a, yy, tt = p[ix], y[ix], t[ix]; t1 = time.perf_counter()
qini_auc_score(y_true=yy, uplift=a, treatment=tt); t2 = time.perf_counter()
uplift_auc_score(y_true=yy, uplift=a, treatment=tt); t3 = time.perf_counter()
uplift_at_k(y_true=yy, uplift=a, treatment=tt, strategy='overall', k=.1); t4 = time.perf_counter()
one = (t1 - t0) + (t2 - t1) + (t3 - t2) + (t4 - t3)
print(f'rows {n:,}; index {t1-t0:.2f}s qini {t2-t1:.2f}s auuc {t3-t2:.2f}s uplift10 {t4-t3:.2f}s')
rep = 6 * ((t2 - t1) + (t3 - t2) + (t4 - t3)) + 3 * (t1 - t0)
print(f'one replicate about {rep:.1f}s; 1000 replicates about {rep*1000/3600:.1f} h on one process')
