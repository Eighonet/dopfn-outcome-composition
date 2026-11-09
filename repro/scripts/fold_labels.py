"""Rebuild group-safe fold labels for LZD and report a hash, to see which versions change them."""
import hashlib, os, sys
code = os.environ['CODE']; sys.path.insert(0, code)
import numpy as np, sklearn
from pathlib import Path
import fm_benchmark as fm
import make_group_split_cache as mk
root = Path(os.environ['DATA_WIN'])
X, t, y, groups, feats = fm.load(root, sys.argv[1])
labels = mk.compute_labels(t, y, groups, fm)
print(f'sklearn {sklearn.__version__} numpy {np.__version__}: labels sha {hashlib.sha256(labels.tobytes()).hexdigest()[:14]} '
      f'holdout {(labels==-1).sum()} fold sizes {[int((labels==f).sum()) for f in range(5)]}', flush=True)
