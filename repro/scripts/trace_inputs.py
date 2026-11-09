"""List the artifacts/local files a runner opens, to pack exactly those for a DataSphere job.

Usage (from repro/rerun): trace_inputs.py tools/run_x.py [runner args...]
Prints each opened file under artifacts/local once, then a size summary per top-level folder.
"""
import os
import runpy
import sys
from collections import defaultdict
from pathlib import Path

ROOT = (Path.cwd() / "artifacts" / "local").resolve()
seen = {}


def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        try:
            path = Path(os.fsdecode(args[0])).resolve()
        except (OSError, ValueError):
            return
        if path.is_relative_to(ROOT) and path not in seen and path.is_file():
            seen[path] = path.stat().st_size


sys.addaudithook(hook)
script, sys.argv = sys.argv[1], sys.argv[1:]
sys.path.insert(0, str(Path(script).parent))
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exit_:
    print("runner exit:", exit_.code, file=sys.stderr)
finally:
    folders = defaultdict(int)
    for path, size in sorted(seen.items()):
        rel = path.relative_to(ROOT)
        print("OPENED", rel.as_posix(), size)
        folders[rel.parts[0]] += size
    for folder, size in sorted(folders.items()):
        print(f"FOLDER {folder} {size / 2**20:.1f} MiB")
