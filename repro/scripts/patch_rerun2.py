"""Let the CS01 multi-dataset runner and its launcher take the device and batch from the environment."""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def edit(name, pairs):
    path = root / name
    text = path.read_bytes().decode("utf-8")
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new)
    path.write_bytes(text.encode("utf-8"))


edit("projects/context_selection/src/multi_dataset_context.py", [
    ('choices=("cpu", "mps", "auto"), default="cpu"', 'choices=("cpu", "mps", "auto", "cuda"), default="cpu"'),
    ('p.add_argument("--batch", type=int, default=2)',
     'p.add_argument("--batch", type=int, default=int(os.environ.get("REPRO_BATCH", "2")))'),
])
edit("tools/run_cs01_e02.py", [
    ('"--device", "cpu", "--folds", "1",', '"--device", os.environ.get("REPRO_DEVICE", "cpu"), "--folds", "1",'),
])
print("patched")
