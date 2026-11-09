"""Map which GPU configurations this DataSphere project can get: one probe per type, cancelled once a VM is assigned.

Usage: python ds/probe_map.py <job> [<job> ...]   (main venv). Prints per probe: allocated type or the error.
"""
import json
import subprocess
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import ds


def get(job_id):
    """Status, assigned type and details through the SDK client (the CLI hides status_details)."""
    code = ("import os,sys,json; sys.path.insert(0, r'%s'); import iam; os.environ['YC_IAM_TOKEN']=iam.iam_token();"
            "from datasphere.client import Client; j=Client().get('%s');"
            "print(json.dumps([j.status, j.actual_cloud_instance_type.name, j.status_details]))") % (ds.DS, job_id)
    out = subprocess.run([str(ds.REPRO / ".venv-ds/Scripts/python.exe"), "-c", code], capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


names = sys.argv[1:]
ds.submit(names)
runs = ds.runs()
pending = {n: runs[n]["job_id"] for n in names if runs[n].get("job_id")}
result = {}
deadline = time.time() + 15 * 60
while pending and time.time() < deadline:
    for name, jid in list(pending.items()):
        status, vm, details = get(jid)
        if vm:   # a VM was assigned: record and cancel at once
            ds.cli(["project", "job", "cancel", "--id", jid], capture_output=True, text=True)
            result[name] = f"ALLOCATED {vm} (cancelled)"
            del pending[name]
        elif status in (4, 5, 6):
            result[name] = f"status {status}: {details}"
            del pending[name]
    time.sleep(15)
for name, jid in pending.items():
    ds.cli(["project", "job", "cancel", "--id", jid], capture_output=True, text=True)
    result[name] = "no VM within 15 min (cancelled)"
for name in names:
    print(f"{name:14s} {result.get(name, 'not created')}", flush=True)
