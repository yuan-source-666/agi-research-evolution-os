# -*- coding: utf-8 -*-
"""增量同步 v5.x 成果到云端归档 /public/home/[REDACTED-CLUSTER-USER]/111111/local_scnet_v8/"""
import os, base64, requests, hashlib

BASE = "https://[REDACTED-ENDPOINT]:58043/jupyter-forward/[REDACTED-ID]"
TOKEN = "[REDACTED-TOKEN]"
LOCAL_ROOT = r"[LOCAL-WORKSPACE]\teleagent\.temp\scnet_v8"
REMOTE_ROOT = "root/private_data/111111/local_scnet_v8"

for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

s = requests.Session()

def api(method, path, **kw):
    return s.request(method, BASE + "/api/contents/" + path,
                     params={"token": TOKEN}, timeout=180, **kw)

FILES = [
    "chat_relay.py",            # v5.6 R1范式+数字守卫核心
    "chat_relay_v4.py.bak3",    # v4 备份（欲望-恐惧版）
    "check_kernel.py",          # 内核模型状态诊断
    "probe_model.py",           # 模型保真度最小化实验
    "probe_model2.py",          # 模型体检（tokenizer/算术/config）
    "RESTORE.md",               # 恢复文档（本地最新版）
]

ok, fail = [], []
for fn in FILES:
    lp = os.path.join(LOCAL_ROOT, fn)
    data = open(lp, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    r = api("PUT", REMOTE_ROOT + "/" + fn,
            json={"content": base64.b64encode(data).decode(),
                  "format": "base64", "type": "file",
                  "path": REMOTE_ROOT + "/" + fn})
    if r.status_code in (200, 201):
        ok.append((fn, sha, len(data)))
    else:
        fail.append((fn, r.status_code, r.text[:100]))

print("uploaded:", len(ok), "failed:", len(fail))
for f in fail:
    print("FAIL:", f)
for fn, sha, n in ok:
    print("OK %-24s %s %d bytes" % (fn, sha[:16], n))
