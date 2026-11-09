"""Put the 12 prior LZD eval-ID files that CS01/E04 checks in place, byte-identical to the archive.

The rerun wrote its CSVs with CRLF on Windows; the archived hashes are of LF files. The CS01/E02 split
files and the E01/v003 folds already exist and only need LF. The single-split fold of E01 v001/v002
(and the two 2026-10-16 legacy runs) is regenerated with the source's own split_indices(folds=1).
Run from repro/rerun with the main venv. Prints OK/BAD per file; writes nothing that fails its hash.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
REPRO = ROOT.parent
INDEX = ROOT / "projects/context_selection/hypotheses/CS01/experiments/E04/v001_prior_eval_sources.csv"
V003 = REPRO / "runs/cs01_e01_v003/20261107T082610_642735Z/LZD"
sys.path.insert(0, str(ROOT / "projects/context_selection/src"))
import multi_dataset_context as mdc


def lf_bytes(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def single_split_eval():
    sys.argv = ["x", "--audit", "--folds", "1", "--split-seed", "42", "--eval-max", "4096",
                "--data-root", str(ROOT / "artifacts/local/data_A_cleaned"), "--datasets", "LZD"]
    args = mdc.parse_args()
    x, t, y, ids, groups, _ = mdc.load_dataset("LZD", args)
    (fold, train, test), = list(mdc.split_indices(x, t, y, groups, args))
    return pd.DataFrame({"source_row": ids[test], "role": "eval"}).to_csv(index=False, lineterminator="\n").encode()


rows = pd.read_csv(INDEX)
single = None
for row in rows.itertuples():
    target = ROOT / row.local_path
    parts = Path(row.local_path).parts
    if parts[2] == "cs01_e02_results":
        data = lf_bytes(target)
    elif parts[3] == "20261013T202852_574199Z":
        data = lf_bytes(V003 / parts[5] / "eval_rows.csv")
    else:
        single = single if single is not None else single_split_eval()
        data = single
    ok = len(data) == row.size and hashlib.sha256(data).hexdigest() == row.sha256
    print("OK " if ok else "BAD", row.local_path, flush=True)
    if ok and (not target.is_file() or target.read_bytes() != data):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
