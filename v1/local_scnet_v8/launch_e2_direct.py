#!/usr/bin/env python3
"""上传修复后的 phase_e2_struct.py + phase_e1_lora.py：
E2-STRUCT 立即启动（GPU 已空），修复版 E1 排在其后自动接力。"""
import os, sys, base64, json, requests

BASE = os.environ.get(
    "JUPYTER_BASE",
    "https://[REDACTED-ENDPOINT]:58043/jupyter-forward/[REDACTED-ID]")
TOKEN = os.environ.get("JUPYTER_TOKEN", "[REDACTED-TOKEN]")

for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"


def api_put(path, content_b64):
    r = requests.put(BASE + "/api/contents" + path, params={"token": TOKEN},
                     json={"content": content_b64, "format": "base64",
                           "type": "file"}, timeout=120)
    r.raise_for_status()


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


for local, remote in [
        (r"[LOCAL-WORKSPACE]\teleagent\.temp\scnet_v8\phase_e2_struct.py",
         "/root/private_data/phase_e2_struct.py"),
        (r"[LOCAL-WORKSPACE]\teleagent\.temp\scnet_v8\phase_e1_lora.py",
         "/root/private_data/phase_e1_lora.py")]:
    b64 = base64.b64encode(open(local, "rb").read()).decode()
    api_put(remote, b64)
    print("uploaded", remote)

# E2-STRUCT 直接启动（GPU 空），exec 保证 PID 不变；修复版 E1 等 E2 结束后接力
code = (
    "import subprocess\n"
    "log2 = open('/root/private_data/phase_e2_struct.log', 'w')\n"
    "p2 = subprocess.Popen(['bash', '-c',\n"
    "  'cd /root/private_data && exec python3 phase_e2_struct.py'],\n"
    "  stdout=log2, stderr=subprocess.STDOUT, start_new_session=True)\n"
    "print('e2_struct pid', p2.pid)\n"
    "wrapper = ('while kill -0 %d 2>/dev/null; do sleep 30; done; "
    "sleep 30; cd /root/private_data && python3 phase_e1_lora.py "
    "> phase_e1_lora.log 2>&1') % p2.pid\n"
    "logw = open('/root/private_data/phase_e1_wrapper2.log', 'w')\n"
    "pw = subprocess.Popen(['bash', '-c', wrapper], stdout=logw,\n"
    "                      stderr=subprocess.STDOUT, start_new_session=True)\n"
    "print('e1 rerun wrapper pid', pw.pid)")
print(run_code(code))
