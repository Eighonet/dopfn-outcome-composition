"""CPU feasibility check: can Do-PFN's internal target normalisation be overridden, and what does it do?

Takes saved Criteo CV contexts (fold 0, seed 42, K=1024), scores the first N eval rows on CPU and
compares (a) the unpatched call with the saved GPU predictions, (b) calls where the z-scoring statistics
of the target are replaced by those of another outcome share. Saves per-arm predictions.
Not an experiment: one context per design, a few hundred rows.
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
SMOKE = REPRO / "home/projects/smiles/cpu_smoke"
sys.path.insert(0, str(SMOKE))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier

from fm_dense_cache import cached_dense_load, preprocess_cached_dense
from gpu_smoke import dopfn_working_dir, to_numpy

torch.set_num_threads(12)
N_EVAL = int(sys.argv[1])
CELLS = sys.argv[2].split(",")
OUT = Path(sys.argv[3])
REPO = REPRO / "home/projects/Do-PFN"
DATA_ROOT = (H18 / "projects/smiles/data/data_A_cleaned").resolve()
RES = H18 / "projects/smiles/results"
FILES = {"T": RES / "fm_cv/criteo/fold0_dopfn_balanced_treatment_ctx1024_seed42_eval8000_cuda0.predictions.npz",
         "TY": RES / "fm_cv/criteo/fold0_dopfn_balanced_treatment_outcome_ctx1024_seed42_eval8000_cuda0.predictions.npz",
         "20%": RES / "fm_dose/criteo/fold0_dopfn_share20_ctx1024_seed42_eval8000_cuda0.predictions.npz",
         "10%": RES / "fm_dose/criteo/fold0_dopfn_share10_ctx1024_seed42_eval8000_cuda0.predictions.npz",
         "R": RES / "fm_cv/criteo/fold0_dopfn_random_ctx1024_seed42_eval8000_cuda0.predictions.npz"}
SHARE = {"nat": None, "p04": 0.042, "p10": 0.10, "p20": 0.20, "p35": 0.35, "p50": 0.50}

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

out = {"mu0": mu0, "ev": ev}
with dopfn_working_dir(REPO):
    import scripts.transformer_prediction_interface.base as base
    original = base.normalize_data
    for cell in CELLS:
        rows, enc = cell.split(":")
        z = np.load(FILES[rows])
        assert np.array_equal(z["eval_indices"][:N_EVAL], ev)
        ctx, saved = z["context_indices"], z["predictions"][:N_EVAL]
        cx, xev = preprocess_cached_dense(x, ctx, ev, features)
        p = SHARE[enc]

        def patched(data, normalize_positions=-1, return_scaling=False, clip=True, std_only=False, p=p):
            _, (mean, std) = original(data, normalize_positions=normalize_positions, return_scaling=True,
                                      clip=clip, std_only=std_only)
            mean, std = torch.full_like(mean, p), torch.full_like(std, float(np.sqrt(p * (1 - p))))
            data = torch.clip((data - mean) / std, min=-100, max=100)
            return (data, (mean, std)) if return_scaling else data

        base.normalize_data = original if p is None else patched
        mdl = base.DoPFNRegressor()
        mdl.device = "cpu"
        mdl.get_max_free_memory_in_gb = lambda: 64.0  # os.sysconf is missing on Windows
        mdl.fit(np.column_stack([t[ctx].astype(np.float32), cx]).astype(np.float32), y[ctx].astype(np.float32))
        start = time.perf_counter()
        R = {k: {1: [], 0: []} for k in ("mean", "median", "ppos", "clip", "tail")}
        for lo in range(0, len(ev), 128):
            a = xev[lo:lo + 128]
            for arm in (1, 0):
                q = torch.from_numpy(np.column_stack([np.full(len(a), arm, dtype=np.float32), a]).astype(np.float32))
                full = mdl.predict_full(q)
                b = full["criterion"].borders.detach().cpu().numpy().astype(np.float64)   # raw outcome units
                pr = np.asarray(full["buckets"], dtype=np.float64).reshape(len(a), -1)
                lo_b, hi_b = b[:-1], b[1:]
                above = np.clip((hi_b - 0.5) / (hi_b - lo_b), 0, 1)                      # bucket share above 0.5
                mid = np.clip((lo_b + hi_b) / 2, 0, 1)
                R["mean"][arm].append(np.asarray(full["mean"], dtype=np.float64).ravel())
                R["median"][arm].append(np.asarray(full["median"], dtype=np.float64).ravel())
                R["ppos"][arm].append(pr @ above)
                R["clip"][arm].append(pr @ mid)
                R["tail"][arm].append(pr[:, 0] + pr[:, -1])
        line = f"{cell:8s} ctx_pos={y[ctx].mean():.3f} sec={time.perf_counter() - start:5.0f}"
        for k in R:
            y1, y0 = np.concatenate(R[k][1]), np.concatenate(R[k][0])
            out[f"{cell}|{k}|y1"], out[f"{cell}|{k}|y0"] = y1, y0
            if k != "tail":
                line += (f" | {k}: rho_tau={spearmanr(y1 - y0, mu0).statistic:+.2f} rho_y1={spearmanr(y1, mu0).statistic:+.2f}"
                         f" rho_y0={spearmanr(y0, mu0).statistic:+.2f} m1={y1.mean():+.3f} m0={y0.mean():+.3f}")
            else:
                line += f" | tail mass y1={y1.mean():.4f} y0={y0.mean():.4f}"
        tau = out[f"{cell}|mean|y1"] - out[f"{cell}|mean|y0"]
        line += f" | vs saved: rho={spearmanr(tau, saved).statistic:+.3f}"
        print(line, flush=True)
        del mdl
    base.normalize_data = original
np.savez_compressed(OUT, **out)
print("saved", OUT)
