import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
REPRO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO / "home/projects/smiles/cpu_smoke"))
import numpy as np, torch
from gpu_smoke import dopfn_working_dir
from typing import Optional, Callable, Union
import torch.nn.modules.transformer as trans
trans.Optional, trans.Callable, trans.Union = Optional, Callable, Union
with dopfn_working_dir(REPRO / "home/projects/Do-PFN"):
    import scripts.transformer_prediction_interface.base as base
    m = base.DoPFNRegressor(); m.device = "cpu"
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 5)).astype(np.float32); X[:, 0] = rng.integers(0, 2, 64)
    m.fit(X, rng.normal(size=64).astype(np.float32))
    crit = m.model_processed_.criterion
    b = crit.borders.detach().cpu().numpy()
    print(type(crit).__name__, "n_borders", len(b))
    print("first 6", np.round(b[:6], 3)); print("last 6", np.round(b[-6:], 3))
    w = np.diff(b)
    print("bucket widths min/median/max", w.min().round(4), np.median(w).round(4), w.max().round(3))
    for p in (0.042, 0.057, 0.10, 0.20, 0.35, 0.50, 0.75):
        lo, hi = -np.sqrt(p / (1 - p)), np.sqrt((1 - p) / p)
        il, ih = np.searchsorted(b, lo) - 1, np.searchsorted(b, hi) - 1
        print(f"share {p:.3f}: z0={lo:+.2f} bucket {il}/{len(b)-1}   z1={hi:+.2f} bucket {ih}/{len(b)-1}  (above last border: {hi > b[-1]})")
    cfg = m.c_processed_
    for k in ("emsize", "nlayers", "nhead", "max_num_classes", "num_buckets", "bptt", "seq_len_used", "max_eval_pos", "num_features"):
        if k in cfg: print(k, cfg[k])
    print("n_params", sum(p.numel() for p in m.model_processed_.parameters()))
