#!/usr/bin/env python3
"""P0 最小修复后重传 evo_daemon.py 到远端（daemon 尚未启动，覆盖即可，无需重启）。"""
import os, base64, requests

BASE = os.environ.get(
    "JUPYTER_BASE",
    "https://REDACTED_JUPYTER_BASE")
TOKEN = os.environ.get("JUPYTER_TOKEN", "REDACTED_JUPYTER_TOKEN")
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"

b64 = base64.b64encode(open(
    r"E:\ai_agent_workspace\teleagent\.temp\scnet_v8\evo_daemon.py",
    "rb").read()).decode()
r = requests.put(BASE + "/api/contents/root/private_data/evo_daemon.py",
                 params={"token": TOKEN},
                 json={"content": b64, "format": "base64", "type": "file"},
                 timeout=120)
r.raise_for_status()
print("uploaded /root/private_data/evo_daemon.py (P0 fixed)")

# 验证远端文件确实是新版：检查 STATE_PATH 关键字
import urllib.request
req = requests.get(BASE + "/api/contents/root/private_data/evo_daemon.py",
                   params={"token": TOKEN, "format": "base64"}, timeout=60)
import json as _json
content = _json.loads(req.text)
raw = base64.b64decode(content["content"]).decode()
for kw in ("STATE_PATH", "FastMemBlk", "MoEBlk", "Hebbian",
           "WIDTH GROW", "'width': D", "growth", "参数自膨胀"):
    print(kw, "->", "OK" if kw in raw else "MISSING")
