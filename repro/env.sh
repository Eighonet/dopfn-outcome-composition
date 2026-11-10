# Runs the unmodified archive code against a project-local home.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; export R
winpath(){ cygpath -w "$1" 2>/dev/null || echo "$1"; }   # native Windows form under Git Bash, unchanged elsewhere
export USERPROFILE="$(winpath "$R/home")" HOME="$R/home"
export HF_HOME="$R/home/.cache/huggingface" PY="$R/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$R/.venv/bin/python"
export CODE="$R/home/projects/smiles/cpu_smoke" PYTHONIOENCODING=utf-8 PYTHONUTF8=1
