"""Make every loop body that the batch patch touched use REPRO_BATCH, not a literal 2."""
import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = subprocess.run(["git", "grep", "-l", "REPRO_BATCH)", "--", "tools/*.py", "projects/*/src/*.py"],
                       cwd=root, capture_output=True, text=True, check=True).stdout.split()
rules = [
    (r"\b(lo|low)\s*\+\s*2\b", r"\1 + REPRO_BATCH"),
    (r"\bj\s*:\s*j\s*\+\s*2\b", "j:j + REPRO_BATCH"),
    (r"\blo\s*\+\s*j\s*\+\s*2\b", "lo + j + REPRO_BATCH"),
]
total = 0
for name in names:
    path = root / name
    lines = path.read_bytes().decode("utf-8").split("\n")
    marks = [i for i, line in enumerate(lines) if "REPRO_BATCH)" in line]
    changed = 0
    for i in marks:
        for k in range(max(0, i - 2), min(len(lines), i + 5)):
            new = lines[k]
            for pattern, repl in rules:
                new = re.sub(pattern, repl, new)
            if new != lines[k]:
                lines[k] = new
                changed += 1
    if changed:
        path.write_bytes("\n".join(lines).encode("utf-8"))
        total += changed
        print(f"{name}: {changed} lines")
print("total", total)
