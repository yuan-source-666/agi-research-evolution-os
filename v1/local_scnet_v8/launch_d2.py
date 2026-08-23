#!/usr/bin/env python3
"""上传 phase_d2_real.py 到 SCNet 并后台启动（D2 真实任务主线）。"""
import os, sys, base64, json, requests

BASE = os.environ.get(
    "JUPYTER_BASE",
    "https://[REDACTED-ENDPOINT]:58043/jupyter-forward/[REDACTED-ID]")
TOKEN = os.environ.get("JUPYTER_TOKEN", "[REDACTED-TOKEN]")
LOCAL = r"[LOCAL-WORKSPACE]\teleagent\.temp\scnet_v8\phase_d2_real.py"
REMOTE = "/root/private_data/phase_d2_real.py"

for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"


def api_put(path, content_b64):
    r = requests.put(BASE + "/api/contents" + path, params={"token": TOKEN},
                     json={"content": content_b64, "format": "base64",
                           "type": "file"}, timeout=120)
    r.raise_for_status()


def run_code(code, timeout=120):
    import uuid, websocket
    r = requests.post(BASE + "/api/kernels", params={"token": TOKEN},
                      json={"name": "python3"}, timeout=30)
    r.raise_for_status()
    kid = r.json()["id"]
    ws_url = BASE.replace("https://", "wss://", 1) + \
        "/api/kernels/%s/channels?token=%s" % (kid, TOKEN)
    ws = websocket.create_connection(ws_url, timeout=timeout)
    msg_id = uuid.uuid4().hex
    req = {"header": {"msg_id": msg_id, "username": "agent", "session": msg_id,
                      "msg_type": "execute_request", "version": "5.3"},
           "parent_header": {}, "metadata": {}, "buffers": [],
           "content": {"code": code, "silent": False, "store_history": False,
                       "user_expressions": {}, "allow_stdin": False,
                       "stop_on_error": True},
           "channel": "shell"}
    ws.send(json.dumps(req))
    out = []
    while True:
        try:
            raw = ws.recv()
        except Exception as e:
            out.append("\n[WS closed: %s]" % e)
            break
        msg = json.loads(raw)
        ch, mt = msg.get("channel"), msg.get("msg_type")
        c = msg.get("content", {})
        if ch == "iopub":
            if mt == "stream":
                out.append(c.get("text", ""))
            elif mt == "error":
                out.append("\n[ERROR] %s: %s" % (c.get("ename"), c.get("evalue")))
            elif mt in ("execute_result", "display_data"):
                d = c.get("data", {})
                if "text/plain" in d:
                    out.append(d["text/plain"])
        elif ch == "shell" and mt in ("execute_reply", "error"):
            if mt == "error":
                out.append("\n[REPLY ERROR] %s: %s" % (c.get("ename"), c.get("evalue")))
            break
    ws.close()
    try:
        requests.delete(BASE + "/api/kernels/" + kid,
                        params={"token": TOKEN}, timeout=10)
    except Exception:
        pass
    return "".join(out)


print("== upload ==")
b64 = base64.b64encode(open(LOCAL, "rb").read()).decode()
api_put(REMOTE, b64)
print("uploaded", REMOTE)

print("== launch ==")
code = (
    "import subprocess, os\n"
    "d = '/root/private_data'\n"
    "log = open(os.path.join(d, 'phase_d2_real.log'), 'w')\n"
    "p = subprocess.Popen(['python3', 'phase_d2_real.py'], cwd=d,\n"
    "                     stdout=log, stderr=subprocess.STDOUT,\n"
    "                     start_new_session=True)\n"
    "print('pid', p.pid)")
print(run_code(code))
print("log: /root/private_data/phase_d2_real.log")
print("result: /root/private_data/phase_d2_real.json")
