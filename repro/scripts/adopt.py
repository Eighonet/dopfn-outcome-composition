"""Put a rerun result in place of the archived one inside the rerun clone, under the archived run id.

Usage: adopt.py <snapshot parent> <new id> <archived id> [<local results parent>]

The archived snapshot stays untouched in SMILES-HypeCheck for comparison. Here it is replaced by the
rerun, every path that names the new id is renamed, and the hashes in local_files.csv and
manifest.json are recomputed from disk, so later runners verify the rerun outputs strictly.
"""
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "rerun"
TEXT = {".md", ".json", ".csv", ".txt"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    subprocess.run(["git", *args], cwd=ROOT, check=True)


def main():
    parent, new_id, old_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    import re
    run_id = re.compile(r"\d{8}T\d{6}_\d+Z")
    assert run_id.fullmatch(new_id) and run_id.fullmatch(old_id) and new_id != old_id,         f"refusing to adopt: ids must be two different run ids, got {new_id!r} and {old_id!r}"
    local_parent = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    snap_new, snap_old = ROOT / parent / new_id, ROOT / parent / old_id
    resuming = not snap_new.is_dir() and snap_old.is_dir()   # an earlier attempt already moved it
    assert snap_new.is_dir() or resuming, f"missing new snapshot {snap_new}"
    if local_parent:
        new_local, old_local = ROOT / local_parent / new_id, ROOT / local_parent / old_id
        if new_local.is_dir():
            assert not old_local.exists(), f"archived local id already taken: {old_local}"
            new_local.rename(old_local)
        assert old_local.is_dir(), f"missing local run {old_local}"
    if not resuming:
        if snap_old.exists():
            git("rm", "-r", "-q", "--", str((parent / old_id).as_posix()))
        snap_new.rename(snap_old)

    small_local = []
    if local_parent:   # protocol.json and status.json of the local run also carry the run id
        small_local = [p for p in (ROOT / local_parent / old_id).rglob("*")
                       if p.is_file() and p.suffix in {".json", ".txt"} and p.stat().st_size < 1_000_000]
    for path in [*snap_old.rglob("*"), *small_local]:
        if path.is_file() and path.suffix in TEXT:
            text = path.read_bytes().decode("utf-8")
            if new_id in text:
                path.write_bytes(text.replace(new_id, old_id).encode("utf-8"))

    local_list = snap_old / "local_files.csv"
    if local_list.is_file():
        rows = list(csv.DictReader(io.StringIO(local_list.read_bytes().decode("utf-8"))))
        for row in rows:
            target = ROOT / (row["path"] if "path" in row else row["local_path"])
            assert target.is_file(), f"listed local file is missing: {target}"
            row["size"], row["sha256"] = str(target.stat().st_size), sha(target)
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        local_list.write_bytes(out.getvalue().encode("utf-8"))

    manifest_path = snap_old / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
        for item in manifest.get("files", []):
            target = snap_old / item["path"]
            if target.is_file():
                item["size"], item["sha256"] = target.stat().st_size, sha(target)
        manifest_path.write_bytes((json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))

    if __import__("os").environ.get("ADOPT_NO_COMMIT") == "1":   # runners check the commit while they run
        print(f"adopted {new_id} -> {old_id} (not committed yet)")
        return
    git("add", "-A", "--", parent.as_posix())
    git("-c", "user.name=repro", "-c", "user.email=repro@local", "commit", "-q", "-m",
        f"Adopt rerun {new_id} as {parent.as_posix()}/{old_id}\n\n"
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    print(f"adopted {new_id} -> {old_id}")


if __name__ == "__main__":
    main()
