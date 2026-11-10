"""CPU pilot: does Do-PFN's effect score depend on how the outcome and the treatment are coded?

Usage: dopfn_flip_smoke.py N_EVAL CELLS OUT [FOLD SEED]   (cells like "T:swap,R:flip"; default context fold 0, seed 42)
Saved Criteo contexts at K=1024 and the first N eval rows of that fold.
`flip` recodes the outcome in the context as 1 - y (the rare class becomes 0), `swap` recodes the treatment as 1 - t,
`nat` changes nothing. Predictions are mapped back to the original outcome and arm labels before saving, so every cell
is comparable with the `nat` cells of dopfn_norm_smoke_2048.npz. One context per design: a pilot, not an experiment.

The Criteo file is not shuffled and the eval indices are sorted: the first 2,048 rows of fold 0 are all treated, so
only N_EVAL = 8000 gives a Qini. The rho printed below uses a risk model fitted on the first 150,000 holdout control
rows, one low-base-rate segment; recheck_pilots.py re-scores every cell against the main risk model.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import time
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
H18 = REPRO / "home18"
os.environ["USERPROFILE"] = str(H18)
os.environ["HOME"] = str(H18)
sys.path.insert(0, str(REPRO / "home/projects/smiles/cpu_smoke"))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier

from fm_dense_cache import cached_dense_load, preprocess_cached_dense
from gpu_smoke import dopfn_working_dir

torch.set_num_threads(12)
N_EVAL = int(sys.argv[1])
CELLS = sys.argv[2].split(",")
OUT = Path(sys.argv[3]).resolve()   # the Do-PFN import below changes the working directory
REPO = REPRO / "home/projects/Do-PFN"
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
RES = H18 / "projects/smiles/results"
FOLD, SEED = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 42)   # another saved context


def saved(sub, strategy):
    return RES / f"{sub}/criteo/fold{FOLD}_dopfn_{strategy}_ctx1024_seed{SEED}_eval8000_cuda0.predictions.npz"


FILES = {"T": saved("fm_cv", "balanced_treatment"), "TY": saved("fm_cv", "balanced_treatment_outcome"),
         "20%": saved("fm_dose", "share20"), "10%": saved("fm_dose", "share10"), "R": saved("fm_cv", "random")}

x, t, y, _, features = cached_dense_load(DATA_ROOT)
t, y = np.asarray(t).astype(int), np.asarray(y).astype(int)
first = np.load(FILES["T"])
ev = first["eval_indices"][:N_EVAL]
hold = first["final_holdout_indices"]
r = hold[t[hold] == 0][:150000]
mu0 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=0).fit(
    np.asarray(x[r], dtype=np.float64), y[r]).predict_proba(np.asarray(x[ev], dtype=np.float64))[:, 1]
print("eval rows", len(ev), "mu0 ready", flush=True)

from typing import Optional, Callable, Union
import torch.nn.modules.transformer as trans
trans.Optional, trans.Callable, trans.Union = Optional, Callable, Union

out = {"mu0": mu0, "ev": ev, "y": y[ev], "t": t[ev]}
with dopfn_working_dir(REPO):
    import scripts.transformer_prediction_interface.base as base
    for cell in CELLS:
        rows, code = cell.split(":")
        z = np.load(FILES[rows])
        assert np.array_equal(z["eval_indices"][:N_EVAL], ev)
        ctx, saved = z["context_indices"], z["predictions"][:N_EVAL]
        cx, xev = preprocess_cached_dense(x, ctx, ev, features)
        flip, swap = "flip" in code, "swap" in code
        tc = 1 - t[ctx] if swap else t[ctx]
        yc = 1 - y[ctx] if flip else y[ctx]
        mdl = base.DoPFNRegressor()
        mdl.device = "cpu"
        mdl.get_max_free_memory_in_gb = lambda: 64.0  # os.sysconf is missing on Windows
        mdl.fit(np.column_stack([tc.astype(np.float32), cx]).astype(np.float32), yc.astype(np.float32))
        start = time.perf_counter()
        R = {k: {1: [], 0: []} for k in ("mean", "median")}
        for lo in range(0, len(ev), 128):
            a = xev[lo:lo + 128]
            for arm in (1, 0):                                        # arm in the original labels
                col = 1 - arm if swap else arm
                q = torch.from_numpy(np.column_stack([np.full(len(a), col, dtype=np.float32), a]).astype(np.float32))
                full = mdl.predict_full(q)
                for k in R:
                    v = np.asarray(full[k], dtype=np.float64).ravel()
                    R[k][arm].append(1 - v if flip else v)            # back to the original outcome
        line = f"{cell:10s} ctx_pos={yc.mean():.3f} ctx_treated={tc.mean():.3f} sec={time.perf_counter() - start:5.0f}"
        for k in R:
            y1, y0 = np.concatenate(R[k][1]), np.concatenate(R[k][0])
            out[f"{cell}|{k}|y1"], out[f"{cell}|{k}|y0"] = y1, y0
            line += (f" | {k}: rho_tau={spearmanr(y1 - y0, mu0).statistic:+.2f} mean_tau={(y1 - y0).mean():+.4f}"
                     f" m1={y1.mean():+.3f} m0={y0.mean():+.3f}")
        tau = out[f"{cell}|mean|y1"] - out[f"{cell}|mean|y0"]
        line += f" | vs saved natural run: rho={spearmanr(tau, saved).statistic:+.3f}"
        print(line, flush=True)
        np.savez_compressed(OUT, **out)
        del mdl
print("saved", OUT)
