"""IAM token for DataSphere CLI calls, made from the service-account authorized key. Never prints secrets.

  python ds/iam.py check     exchange the key for a token and report only that it worked
  python ds/iam.py projects  list the clouds, DataSphere communities and projects this account can see
The key file comes from YC_SA_KEY_FILE, default %USERPROFILE%/.yc/authorized_key.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import grpc
import jwt
import requests
from datasphere.api import iam_endpoint
from datasphere.auth import CreateIamTokenRequest, IamTokenServiceStub

KEY = Path(os.environ.get("YC_SA_KEY_FILE", r"C:\Users\Eighonet Aerta\.yc\authorized_key.json"))
_cache = {}


def iam_token() -> str:
    if _cache.get("expires", 0) > time.time() + 300:
        return _cache["token"]
    key = json.loads(KEY.read_text())
    pem = key["private_key"][key["private_key"].index("-----BEGIN"):]   # Yandex keys start with a notice line
    now = int(time.time())
    signed = jwt.encode({"aud": "https://iam.api.cloud.yandex.net/iam/v1/tokens", "iss": key["service_account_id"],
                         "iat": now, "exp": now + 3600}, pem, algorithm="PS256", headers={"kid": key["id"]})
    stub = IamTokenServiceStub(grpc.secure_channel(iam_endpoint, grpc.ssl_channel_credentials()))
    _cache.update(token=stub.Create(CreateIamTokenRequest(jwt=signed)).iam_token, expires=now + 11 * 3600)
    return _cache["token"]


def get(url, **params):
    r = requests.get(url, params=params, headers={"Authorization": f"Bearer {iam_token()}"}, timeout=30)
    return r.status_code, (r.json() if r.content else {})


def projects():
    code, clouds = get("https://resource-manager.api.cloud.yandex.net/resource-manager/v1/clouds")
    print("clouds:", code, [(c["name"], c["id"], c.get("organizationId")) for c in clouds.get("clouds", [])])
    for org in {c.get("organizationId") for c in clouds.get("clouds", []) if c.get("organizationId")}:
        code, comm = get("https://datasphere.api.cloud.yandex.net/datasphere/v2/communities", organizationId=org)
        print("communities in", org, code, comm.get("message", ""))
        for c in comm.get("communities", []):
            print(f"  community {c['name']!r} {c['id']}")
            code, proj = get("https://datasphere.api.cloud.yandex.net/datasphere/v2/projects", communityId=c["id"])
            for p in proj.get("projects", []):
                print(f"    project {p['name']!r} {p['id']}")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        print("IAM token issued, length", len(iam_token()))
    else:
        projects()
