"""Resubmit jobs rejected with 'Job execution quota exhausted' as soon as a job slot frees up.

Usage: python ds/resubmit_waiter.py <slots> <job> [<job> ...]   (main venv; only API calls, no computation)
Checks every 2 minutes how many hs_* jobs are active and submits the next pending job while fewer than
<slots> are active. Stops when every pending job is submitted, or after 8 hours.
"""
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import ds

ACTIVE = {"CREATING", "EXECUTING", "UPLOADING_OUTPUT", "CANCELLING", "PREPARING"}


def status(job_id):
    out = ds.cli(["project", "job", "get", "--id", job_id, "--format", "json"], capture_output=True, text=True).stdout
    try:
        return json.loads(out[out.index("{"):]).get("status", "")
    except ValueError:
        return ""


slots, pending = int(sys.argv[1]), sys.argv[2:]
deadline = time.time() + 8 * 3600
while pending and time.time() < deadline:
    runs = ds.runs()
    active = sum(status(r["job_id"]) in ACTIVE for name, r in runs.items() if name.startswith("hs_") and r.get("job_id")
                 and name not in pending)
    if active < slots:
        name = pending.pop(0)
        ds.submit([name])
        time.sleep(60)
        if status(ds.runs()[name]["job_id"]) == "ERROR":   # rejected again (quota shared or not yet freed): keep it queued
            pending.append(name)
            print(time.strftime("%H:%M"), "rejected again", name, flush=True)
            time.sleep(240)
        else:
            print(time.strftime("%H:%M"), "submitted", name, "with", active, "active", flush=True)
        continue
    time.sleep(120)
print(time.strftime("%H:%M"), "waiter done; still pending:", pending, flush=True)
