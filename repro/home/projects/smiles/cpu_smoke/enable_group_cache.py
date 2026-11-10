#!/usr/bin/env python3
"""Patch a compatible benchmark script to use cached group splits."""
from pathlib import Path
import argparse
import ast
import shutil


def replace_once(text, old, new, description):
    number = text.count(old)
    if number != 1:
        raise RuntimeError(f"{description}: expected 1 exact match, found {number}; no files modified")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="fm_benchmark.py")
    args = parser.parse_args()
    target = Path(args.target).resolve()
    helper = target.parent / "fm_group_cache.py"
    if not helper.is_file():
        raise FileNotFoundError(f"First put fm_group_cache.py in the same directory: {helper}")
    orig = target.read_text(encoding="utf-8")
    if "from fm_group_cache import cached_split" in orig:
        raise RuntimeError("Already installed; no changes made")

    modified = replace_once(
        orig,
        "def load(root: Path, name: str):",
        "def load(root: Path, name: str, use_group_hashes=True):",
        "load() signature",
    )
    # Row identifiers are not the grouping keys for these datasets.
    
    modified = replace_once(
        modified,
        'groups = x[key].to_numpy().copy() if cfg["group"] else None',
        'groups = x[key].to_numpy().copy() if cfg["group"] and use_group_hashes else None',
        "epk_id group initialization",
    )
    # Preserve the existing grouping by identical feature vectors.
    original_group_block = '    X = df[features].copy()\n    if cfg["group"]:\n'
    modified = replace_once(
        modified,
        original_group_block,
        '    X = df[features].copy()\n    if cfg["group"] and use_group_hashes:\n',
        "feature-vector group hashing",
    )
    modified = replace_once(
        modified,
        '        X, t, y, groups, features = load(data_root, a.dataset)\n'
        '        tr, ev, final_idx, split_method = split_cv(t, y, groups, a.fold)\n',
        '        use_cache = a.dataset in ("criteo", "lzd")\n'
        '        X, t, y, groups, features = load(\n'
        '            data_root, a.dataset, use_group_hashes=not use_cache\n'
        '        )\n'
        '        if use_cache:\n'
        '            from fm_group_cache import cached_split\n'
        '            tr, ev, final_idx, split_method, cache_sha256 = cached_split(\n'
        '                data_root, a.dataset, len(t), features, a.fold\n'
        '            )\n'
        '            record["split_cache_sha256"] = cache_sha256\n'
        '            print(f"Verified grouped-split cache: {a.dataset}, fold={a.fold}", flush=True)\n'
        '        else:\n'
        '            tr, ev, final_idx, split_method = split_cv(t, y, groups, a.fold)\n',
        "main split selection",
    )
    ast.parse(modified, filename=str(target))
    backup = target.with_name(target.name + ".before_split_cache")
    if backup.exists():
        raise FileExistsError(f"Backup already exists; avoid overwriting: {backup}")
    shutil.copy2(target, backup)
    target.write_text(modified, encoding="utf-8")
    print("OK: grouped split cache activated for Criteo/LZD")
    print("Other datasets and model adapters unchanged")
    print(f"Backup: {backup}")
    print(f"Updated: {target}")


if __name__ == "__main__":
    main()
