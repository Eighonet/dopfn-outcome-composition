"""Adopt a rerun whose local outputs are split into several run folders (one per cohort or split).

Usage: adopt_runs.py <snapshot parent> <new snapshot id> <archived snapshot id> <local results root>

Archived run ids come from the archived local_files.csv in SMILES-HypeCheck. Each new run folder is
matched to the archived one by its parent folder (cohort_404, split_101, ...), renamed to the archived
id, the new ids inside the new snapshot are replaced, and adopt.py finishes the job.
"""
import re
import subprocess
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
RERUN, ORIGINAL = REPRO / "rerun", REPRO / "SMILES-HypeCheck"
RUN_ID = r"\d{8}T\d{6}_\d+Z"


def main():
    parent, new_snap, old_snap, root = sys.argv[1:5]
    listing = (ORIGINAL / parent / old_snap / "local_files.csv").read_text(encoding="utf-8")
    archived = dict(re.findall(rf"{re.escape(root)}/([^/,]+)/({RUN_ID})/", listing))
    assert archived, "no per-run folders found in the archived local_files.csv"
    snapshot = RERUN / parent / new_snap
    renamed = {}
    for sub, old_id in sorted(archived.items()):
        folder = RERUN / root / sub
        fresh = [p for p in folder.iterdir() if p.is_dir() and re.fullmatch(RUN_ID, p.name) and p.name != old_id]
        assert len(fresh) == 1, f"{folder}: expected one new run, found {[p.name for p in fresh]}"
        assert not (folder / old_id).exists(), f"{folder / old_id} already exists"
        fresh[0].rename(folder / old_id)
        renamed[fresh[0].name] = old_id
        print(f"{sub}: {fresh[0].name} -> {old_id}")
    for path in snapshot.rglob("*"):
        if path.is_file() and path.suffix in {".md", ".json", ".csv", ".txt"}:
            text = path.read_bytes().decode("utf-8")
            new = text
            for a, b in renamed.items():
                new = new.replace(a, b)
            if new != text:
                path.write_bytes(new.encode("utf-8"))
    subprocess.run([sys.executable, str(REPRO / "scripts" / "adopt.py"), parent, new_snap, old_snap], check=True)


if __name__ == "__main__":
    main()
