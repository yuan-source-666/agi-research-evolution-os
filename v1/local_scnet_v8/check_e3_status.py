#!/usr/bin/env python3
"""查 E3-ARCH 架构自进化实况：进程 + 日志尾部 + 结果 JSON + GPU"""
import os, json, requests

BASE = os.environ.get(
    "JUPYTER_BASE",
    "https://[REDACTED-ENDPOINT]:58043/jupyter-forward/[REDACTED-ID]")
TOKEN = os.environ.get("JUPYTER_TOKEN", "[REDACTED-TOKEN]")
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"


def run_code(code, timeout=180):
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
        elif ch == "shell" and mt in ("execute_reply", "error"):
            break
    ws.close()
    try:
        requests.delete(BASE + "/api/kernels/" + kid,
                        params={"token": TOKEN}, timeout=10)
    except Exception:
        pass
    return "".join(out)


print(run_code(
    "import subprocess\n"
    "print(subprocess.run(['bash','-lc',\n"
    "  'echo ===PROC===; ps aux | grep phase_e3_arch | grep -v grep; "
    "echo; echo ===LOG===; tail -25 /root/private_data/phase_e3_arch.log 2>/dev/null; "
    "echo; echo ===JSON===; cat /root/private_data/phase_e3_arch.json 2>/dev/null | head -c 600; "
    "echo; echo ===GPU===; nvidia-smi --query-gpu=memory.used --format=csv,noheader'],\n"
    "  capture_output=True, text=True).stdout)"))
