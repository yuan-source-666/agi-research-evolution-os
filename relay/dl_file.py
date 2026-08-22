#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 SCNet 下载文件：python dl_file.py <remote_path> <local_path>"""
import sys
import base64
import requests

BASE = "https://REDACTED_JUPYTER_BASE"
TOKEN = "REDACTED_JUPYTER_TOKEN"

for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    import os
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"


def main():
    remote, local = sys.argv[1], sys.argv[2]
    r = requests.get(BASE + "/api/contents" + remote, params={"token": TOKEN},
                     timeout=180)
    r.raise_for_status()
    data = r.json()
    if data.get("format") == "base64":
        raw = base64.b64decode(data["content"])
    else:
        raw = data["content"].encode("utf-8")
    with open(local, "wb") as f:
        f.write(raw)
    print("downloaded %s -> %s (%d bytes)" % (remote, local, len(raw)))


if __name__ == "__main__":
    main()
