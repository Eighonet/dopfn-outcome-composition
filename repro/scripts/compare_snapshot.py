"""Compare every CSV that a rerun snapshot shares with the archived one; print the largest differences.

Usage: compare_snapshot.py <rerun snapshot dir> <archived snapshot dir> [<report file>]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SKIP = {"local_files.csv", "source_files.csv", "bootstrap_draws.csv", "pooled_bootstrap_draws.csv"}


def compare(new_dir, old_dir):
    lines, worst = [], 0.0
    for old in sorted(Path(old_dir).glob("*.csv")):
        new = Path(new_dir) / old.name
        if old.name in SKIP or not new.is_file():
            continue
        a, b = pd.read_csv(old), pd.read_csv(new)
        # identifiers are text and integer columns (dataset, method, seed, fold); measured values are floats
        keys = [c for c in a.columns if c in b.columns and not np.issubdtype(a[c].dtype, np.floating)]
        numeric = [c for c in a.columns if c not in keys and c in b.columns and np.issubdtype(a[c].dtype, np.floating)]
        if not numeric:
            continue
        if keys:
            merged = a.merge(b, on=keys, suffixes=("_old", "_new"), how="outer", indicator=True)
            missing = int((merged["_merge"] != "both").sum())
            merged = merged[merged["_merge"] == "both"]
        else:
            if len(a) != len(b):
                lines.append(f"{old.name}: row count {len(a)} vs {len(b)}")
                continue
            merged = a.add_suffix("_old").join(b.add_suffix("_new"))
            missing = 0
        diffs = {c: float(np.nanmax(np.abs(merged[c + "_new"] - merged[c + "_old"]))) if len(merged) else 0.0
                 for c in numeric}
        big = {c: v for c, v in diffs.items() if v > 1e-6}
        worst = max([worst, *diffs.values()])
        lines.append(f"{old.name}: {len(merged)} rows matched, {missing} unmatched; "
                     + (", ".join(f"{c} max|d|={v:.6g}" for c, v in sorted(big.items(), key=lambda kv: -kv[1])[:6])
                        or "all numbers equal to 1e-6"))
    return lines, worst


if __name__ == "__main__":
    lines, worst = compare(sys.argv[1], sys.argv[2])
    text = "\n".join(lines + [f"largest difference: {worst:.6g}"])
    print(text)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text + "\n", encoding="utf-8")
