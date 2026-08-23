# -*- coding: utf-8 -*-
"""把本地 scnet_v8 全部文件上传到远端归档 [REDACTED-CLUSTER-PATH]/111111/local_scnet_v8/"""
import os, base64, requests, json, hashlib

BASE = "https://REDACTED_JUPYTER_BASE"
TOKEN = "REDACTED_JUPYTER_TOKEN"
LOCAL_ROOT = r"E:\ai_agent_workspace\teleagent\.temp\scnet_v8"
REMOTE_ROOT = "root/private_data/111111/local_scnet_v8"

for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

s = requests.Session()

def api(method, path, **kw):
    r = s.request(method, BASE + "/api/contents/" + path,
                  params={"token": TOKEN}, timeout=180, **kw)
    return r

def mkdirs(rel):
    parts = rel.split("/")
    cur = ""
    for p in parts:
        cur = (cur + "/" + p).strip("/")
        r = api("GET", "root/private_data/" + cur if cur else "")
        # use PUT to ensure folder exists (idempotent)
        api("PUT", "root/private_data/" + cur,
            json={"type": "directory", "path": cur})

uploaded, skipped, failed = 0, 0, []
manifest = []
for dirpath, dirnames, filenames in os.walk(LOCAL_ROOT):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    rel_dir = os.path.relpath(dirpath, LOCAL_ROOT).replace("\\", "/")
    rel_dir = "" if rel_dir == "." else rel_dir
    for fn in filenames:
        if fn in (".chat_kernel_id",) or fn.endswith(".log") and fn.startswith("relay_run"):
            continue
        lp = os.path.join(dirpath, fn)
        rp = (REMOTE_ROOT + ("/" + rel_dir if rel_dir else "")).replace("root/private_data/", "")
        try:
            data = open(lp, "rb").read()
            sha = hashlib.sha256(data).hexdigest()[:16]
            payload = {
                "content": base64.b64encode(data).decode(),
                "format": "base64",
                "type": "file",
                "path": REMOTE_ROOT + ("/" + rel_dir if rel_dir else "") + "/" + fn,
            }
            r = api("PUT", REMOTE_ROOT + ("/" + rel_dir if rel_dir else "") + "/" + fn, json=payload)
            if r.status_code not in (200, 201):
                failed.append((fn, r.status_code, r.text[:100]))
            else:
                uploaded += 1
                manifest.append("%s  %s  %d bytes" % (sha, (rel_dir + "/" if rel_dir else "") + fn, len(data)))
        except Exception as e:
            failed.append((fn, "EXC", str(e)[:100]))

print("uploaded:", uploaded, "failed:", len(failed))
for f in failed:
    print("FAIL:", f)

# 写 manifest
mf = "\n".join(manifest) + "\n"
api("PUT", REMOTE_ROOT + "/LOCAL_MANIFEST.txt",
    json={"content": mf, "format": "text", "type": "file",
          "path": REMOTE_ROOT + "/LOCAL_MANIFEST.txt"})
print("manifest entries:", len(manifest))
