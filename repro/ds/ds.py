"""Run the remaining reproduction and paper experiments as Yandex DataSphere jobs.

  python ds/ds.py packs                 build input archives in ds/packs (code.tar is rebuilt every time)
  python ds/ds.py submit <job|group>... render ds/cfg/<job>.yaml and start each job with --async
  python ds/ds.py status                show every submitted job
  python ds/ds.py fetch <job>...        download outputs and logs to ds/results/<job> and unpack them there

Jobs: probe_cpu probe_gpu, dose_0..dose_9 (group: dose), leak_42..leak_46 (group: leak), e17chain, lzdchain.
The project id comes from DS_PROJECT or ds/project_id.txt. Results are only unpacked under ds/results;
merging them into repro/ is a separate, deliberate step.
"""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

DS = Path(__file__).resolve().parent
REPRO = DS.parent
PROJECT = REPRO.parent                       # causal_fm_project: archive paths are relative to it
CLI = REPRO / ".venv-ds/Scripts/datasphere.exe"
RUNS = DS / "runs.json"
SKIP = {"__pycache__", ".pytest_cache", ".venv", ".venv-dopfn", ".venv-ds"}


def add_tree(tar, source, exclude=()):
    source = Path(source)
    if source.is_file():
        tar.add(source, arcname=source.relative_to(PROJECT).as_posix())
        return
    for top, dirs, files in os.walk(source):
        rel_top = Path(top).relative_to(PROJECT).as_posix()
        dirs[:] = [d for d in dirs if d not in SKIP and f"{rel_top}/{d}" not in exclude]
        for name in files:
            if not name.endswith(".pyc"):
                path = Path(top) / name
                tar.add(path, arcname=path.relative_to(PROJECT).as_posix(), recursive=False)


def traced(log):
    """File list from scripts/trace_inputs.py output, as paths under rerun/artifacts/local."""
    root = REPRO / "rerun/artifacts/local"
    return [root / line.split()[1] for line in (REPRO / "home/projects/smiles/logs/rerun" / log).read_text().splitlines()
            if line.startswith("OPENED ") and not line.split()[1].startswith((".matplotlib", "data_A_cleaned"))]


PACKS = {
    "code": lambda: [REPRO / "scripts", REPRO / "home/projects/smiles/cpu_smoke", REPRO / "archived_from_rerun",
                     REPRO / "rerun", REPRO / "SMILES-HypeCheck"],
    "data": lambda: [PROJECT / "data_A_cleaned"],
    "dopfn": lambda: [REPRO / "home/projects/Do-PFN", REPRO / "home/projects/smiles/models"],
    "cache18": lambda: [REPRO / "home18/projects/smiles/cache"],
    "weights": lambda: [REPRO / "home/.cache/causalpfn", REPRO / "home/.cache/huggingface"],
    "state_e17": lambda: traced("trace_e17.log"),
    "state_lzd": lambda: traced("trace_cs01_e04.log"),
    "state_hs10": lambda: [REPRO / "rerun_compare/hs10_scores.npz"],
    # fm_holdout checks the holdout identity against the completed CV fold-0 results
    "state_hs": lambda: sorted((REPRO / "home18/projects/smiles/results/fm_cv/criteo").glob("fold0_*.json")),
}
# next wave of dependent runners (CPU, batch 2); each state pack holds exactly what their --check opened
DEPS = {"deps_a1": ["m01_e14", "m06_e05"], "deps_a2": ["m06_e04", "m06_e09"], "deps_a3": ["m06_e06", "m06_e14"]}
for _job, _runners in DEPS.items():
    PACKS[f"state_{_job}"] = (lambda rs: lambda: sorted({p for r in rs for p in traced(f"trace_{r}.log")}))(_runners)
EXCLUDE = {"repro/rerun/artifacts/local", "repro/rerun/logs", "repro/SMILES-HypeCheck/artifacts/local",
           "repro/home/projects/Do-PFN/.git"}


def build_packs():
    (DS / "packs").mkdir(exist_ok=True)
    for name, sources in PACKS.items():
        out = DS / "packs" / f"{name}.tar"
        if out.exists() and name != "code":
            print(f"{name}: kept ({out.stat().st_size / 2**20:.0f} MiB)")
            continue
        with tarfile.open(out.with_suffix(".tmp"), "w") as tar:
            for source in sources():
                add_tree(tar, source, EXCLUDE)
        out.with_suffix(".tmp").replace(out)
        size = out.stat().st_size
        assert size < 5 * 2**30, f"{name} exceeds the 5 GB per-file limit"
        print(f"{name}: {size / 2**20:.0f} MiB")
    script = (DS / "job.sh").read_bytes().replace(b"\r\n", b"\n")
    (DS / "packs/job.sh").write_bytes(script)
    (DS / "packs/noop.py").write_text('"""Placeholder local module; the job code comes in code.tar."""\n'
                                     'if __name__ == "__main__":\n    pass\n', newline="\n")


def jobs():
    table = {
        "probe_cpu": dict(kind="probe", env="main", vm=["c1.4"], packs=[]),
        # g1.1 returned "Unable to allocate any VM" twice; DataSphere tries the listed types in order
        "probe_gpu": dict(kind="probe", env="main", vm=["g1.1", "g2.1", "gt4.1", "gt4i.1"], packs=[]),
        "probe_v100": dict(kind="probe", env="main", vm=["g1.1"], packs=[]),
        # one configuration per probe, to map which GPU types this project can get
        **{f"probe_{v.replace('.', '')}": dict(kind="probe", env="main", vm=[v], packs=[])
           for v in ("g1.1", "g1.2", "g1.4", "g2.1", "g2.2", "gt4i.1")},
        "e17chain": dict(kind="e17chain", env="main", vm=["g1.1"], packs=["code", "data", "weights", "state_e17"]),
        "lzdchain": dict(kind="lzdchain", env="main", vm=["g1.1"], packs=["code", "data", "weights", "state_lzd"]),
    }
    # 10 active jobs per quota: e17chain + lzdchain + 6 dose shards (20 Do-PFN runs each) + 1 leakage job
    lines = (REPRO / "home18/projects/smiles/manifests/dose_all.txt").read_text().splitlines()
    for i in range(6):
        part = DS / "packs" / f"dose_{i}.txt"
        if (DS / "packs").is_dir():
            part.write_text("\n".join(lines[i::6]) + "\n", newline="\n")
        table[f"dose_{i}"] = dict(kind="dose", env="dopfn", vm=["g1.1"], packs=["code", "data", "dopfn", "cache18"],
                                  extra={"manifest": f"packs/dose_{i}.txt"})
    table["leak"] = dict(kind="leakage", env="main", vm=["c1.8"], packs=["code", "data"], args={"seed": "42,43,44,45,46"})
    for name, runners in DEPS.items():
        table[name] = dict(kind="runners", env="main", vm=["c1.8"], packs=["code", "data", "weights", f"state_{name}"],
                           args={"runners": ",".join(runners)})
    # paired 20% vs T+Y on the 200k holdout subsample, context seeds 42-51, one GPU type for every run
    for seed in range(42, 52):
        table[f"hs_{seed}"] = dict(kind="holdout_share", env="dopfn", vm=list(HS_VM),
                                   packs=["code", "data", "dopfn", "cache18", "state_hs"],
                                   args={"runs": f"share20:{seed},balanced_treatment_outcome:{seed}"})
    table["hs10boot"] = dict(kind="hs10boot", env="main", vm=["c1.32", "c1.8"], packs=["code", "state_hs10"],
                             args={"draws": 1000})
    return table


# V100/A100 are not allocated in this project (see DEVIATIONS.md); T4 is, with T4i as fallback.
# Both arms of a seed run in the same job, so each paired difference stays on one GPU.
HS_VM = ("gt4.1", "gt4i.1")


def render(name, job):
    inputs = ["  - packs/job.sh: JOBSH"] + [f"  - packs/{p}.tar: {p.upper()}" for p in job["packs"]]
    inputs += [f"  - {path}: {key.upper()}" for key, path in job.get("extra", {}).items()]
    keys = [f"{p}=${{{p.upper()}}}" for p in job["packs"]] + [f"{k}=${{{k.upper()}}}" for k in job.get("extra", {})]
    keys += [f"{k}={v}" for k, v in job.get("args", {}).items()] + ["out=${OUT}"]
    keys = [k.split("=", 1)[0].startswith("state_") and "state=" + k.split("=", 1)[1] or k for k in keys]
    text = "\n".join([
        f"name: {name.replace('_', '-')}", f"desc: causal FM reproduction, {job['kind']}",
        f"cmd: bash ${{JOBSH}} {job['kind']} " + " ".join(keys),
        # fully manual env: version, requirements and a non-empty local-paths (the CLI rejects an empty list);
        # the real code comes in code.tar. The CLI reads no-deps only as the string 'true'.
        "env:", "  python:", "    type: manual", "    version: '3.11'",
        f"    requirements-file: requirements-{job['env']}.txt", "    local-paths:", "      - packs/noop.py",
        # exact local freeze plus torch's Linux CUDA wheels; resolving would move sklearn off 1.5.2
        "    pip:", "      no-deps: 'true'",
        "inputs:", *inputs, "outputs:", f"  - {name}_out.tar.gz: OUT",
        "cloud-instance-types:", *[f"  - {vm}" for vm in job["vm"]], ""])
    (DS / "cfg").mkdir(exist_ok=True)
    path = DS / "cfg" / f"{name}.yaml"
    path.write_text(text, newline="\n")
    return path


def cli(args, **kw):
    """Run the DataSphere CLI with an IAM token from the service-account key (ds/iam.py), never printed."""
    sys.path.insert(0, str(DS))
    token = subprocess.run([str(REPRO / ".venv-ds/Scripts/python.exe"), "-c",
                            "import sys; sys.path.insert(0, r'%s'); import iam; print(iam.iam_token())" % DS],
                           capture_output=True, text=True, check=True).stdout.strip()
    return subprocess.run([str(CLI), *args], env={**os.environ, "YC_IAM_TOKEN": token}, **kw)


def project_id():
    pid = os.environ.get("DS_PROJECT") or (DS / "project_id.txt").read_text().strip()
    assert pid, "set DS_PROJECT or write ds/project_id.txt"
    return pid


def runs():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def submit(names):
    table, done = jobs(), runs()
    expanded = [n for name in names for n in ([k for k in table if k.startswith(name + "_")] or [name])]
    for name in expanded:
        cfg = render(name, table[name])
        info = DS / "cfg" / f"{name}.exec.json"
        cmd = [str(CLI), "project", "job", "execute", "-p", project_id(), "-c", str(cfg.relative_to(DS)),
               "--async", "-o", str(info.relative_to(DS))]
        print(">", " ".join(cmd), flush=True)
        rc = cli(cmd[1:], cwd=DS).returncode
        text = info.read_text().strip() if info.exists() else ""
        record = json.loads(text) if text else {}
        done[name] = dict(rc=rc, **record)
        RUNS.write_text(json.dumps(done, indent=2))
        print(name, "rc", rc, record, flush=True)


def job_id(name):
    record = runs()[name]
    return record.get("job_id") or record.get("id")


def status():
    for name, record in runs().items():
        jid = job_id(name)
        out = cli(["project", "job", "get", "--id", jid, "--format", "json"],
                  capture_output=True, text=True).stdout if jid else ""
        try:
            info = json.loads(out[out.index("{"):])   # the CLI prints log lines before the JSON
            print(f"{name:12s} {jid} {info.get('status')} {info.get('created_at', '')} {info.get('finished_at', '')}")
        except json.JSONDecodeError:
            print(f"{name:12s} {jid} {out.strip()[:120] or record}")


def fetch(names):
    for name in names:
        target = DS / "results" / name
        cli(["project", "job", "download-files", "--id", job_id(name), "--with-logs",
             "--output-dir", str(target)], check=True)
        for archive in target.rglob("*_out.tar.gz"):
            with tarfile.open(archive) as tar:
                tar.extractall(target / "unpacked", filter="data")
        print(name, "->", target)


if __name__ == "__main__":
    action, args = sys.argv[1], sys.argv[2:]
    {"packs": lambda: build_packs(), "submit": lambda: submit(args), "status": status,
     "fetch": lambda: fetch(args)}[action]()
