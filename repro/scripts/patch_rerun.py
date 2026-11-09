"""Make the rerun clone GPU-capable. Without the environment variables behaviour is unchanged."""
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = subprocess.run(["git", "ls-files", "tools/*.py", "projects/*/src/*.py"], cwd=root,
                       capture_output=True, text=True, check=True).stdout.split()
helper = 'import os as _repro_os\nREPRO_BATCH = int(_repro_os.environ.get("REPRO_BATCH", "2"))\n'
device = 'torch.device(_repro_os.environ.get("REPRO_DEVICE", "cpu"))'
guard = re.compile(r"np\.testing\.assert_allclose\((\w+), (\w+), rtol=2e-4, atol=2e-5\)")
counts = dict(files=0, device=0, loops=0, guards=0)

for name in names:
    path = root / name
    text = path.read_bytes().decode("utf-8")
    new, a = re.subn(r'torch\.device\("cpu"\)', device, text)
    new, b = guard.subn(r'print("GUARD", float(np.abs(\1 - \2).max()), flush=True)', new)
    new, c = re.subn(r"range\(0, ([^,()]+(?:\([^()]*\))?), 2\)", r"range(0, \1, REPRO_BATCH)", new)
    new, d = re.subn(r"min\(lo \+ 2,", "min(lo + REPRO_BATCH,", new)
    if not (a or b or c or d):
        continue
    if "_repro_os" in new:
        lines = new.split("\n")
        at = 0
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ")) and "__future__" not in line:
                at = i
                break
        new = "\n".join(lines[:at]) + ("\n" if at else "") + helper + "\n".join(lines[at:])
    path.write_bytes(new.encode("utf-8"))
    counts["files"] += 1
    counts["device"] += a
    counts["guards"] += b
    counts["loops"] += c + d
print(counts)
