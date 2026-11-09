#!/usr/bin/env python3
"""Patch a compatible benchmark script to use the Criteo dense cache."""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path


def replace_once(code, old, new, label):
    count = code.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected 1 match, found {count}; file unchanged')
    return code.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default='fm_benchmark.py')
    a = p.parse_args()
    target = Path(a.target).resolve()
    original = target.read_text(encoding='utf-8')
    if 'from fm_dense_cache import cached_dense_load' in original:
        raise RuntimeError('Dense cache already enabled; no changes made')
    if not (target.parent / 'fm_dense_cache.py').is_file():
        raise FileNotFoundError('Place fm_dense_cache.py next to fm_benchmark.py')

    before = '''        X, t, y, groups, features = load(
            data_root, a.dataset, use_group_hashes=not use_cache
        )
'''
    after = '''        if a.dataset == "criteo":
            from fm_dense_cache import cached_dense_load
            X, t, y, groups, features = cached_dense_load(data_root)
        else:
            X, t, y, groups, features = load(
                data_root, a.dataset, use_group_hashes=not use_cache
            )
'''
    result = replace_once(original, before, after, 'main load routing')
    old_prep = '''        cx, xev = preprocess_train_only(X, ctx, ev)
'''
    new_prep = '''        if a.dataset == "criteo":
            from fm_dense_cache import preprocess_cached_dense
            cx, xev = preprocess_cached_dense(X, ctx, ev, features)
        else:
            cx, xev = preprocess_train_only(X, ctx, ev)
'''
    result = replace_once(result, old_prep, new_prep, 'context-only preprocess routing')
    ast.parse(result, filename=str(target))
    backup = target.with_name(target.name + '.before_dense_cache')
    if backup.exists():
        raise FileExistsError(f'Backup exists: {backup}; no changes made')
    shutil.copy2(target, backup)
    target.write_text(result, encoding='utf-8')
    print('OK: read-only dense Criteo cache enabled; group-split cache unchanged')
    print('Backup:', backup)

if __name__ == '__main__':
    main()
