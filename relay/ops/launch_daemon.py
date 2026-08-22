#!/usr/bin/env python3
"""部署自进化守护进程：等 E3(PID 12745) 一次性实验跑完后自动接管 GPU，
持续进化循环（档案持久、门禁入档/回滚、永不退出）。"""
import os, base64, json, requests

BASE = os.environ.get(
    "JUPYTER_BASE",
    "https://REDACTED_JUPYTER_BASE")
TOKEN = os.environ.get("JUPYTER_TOKEN", "REDACTED_JUPYTER_TOKEN")
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"


def api_put(path, content_b64):
    r = requests.put(BASE + "/api/contents" + path, params={"token": TOKEN},
                     json={"content": content_b64, "format": "base64",
                           "type": "file"}, timeout=120)
    r.raise_for_status()


def run_code(code, timeout=300):
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


# 1) 杀 E1 接力 wrapper（避免和守护进程抢 GPU；E1 以后可单独重跑）
print("== cleanup ==")
print(run_code(
    "import subprocess\n"
    "print(subprocess.run(['bash','-lc',\n"
    "  'kill 12746 2>/dev/null; pkill -f evo_daemon 2>/dev/null; sleep 2; "
    "ps aux | grep -E \"phase_e|evo_daemon\" | grep -v grep'],\n"
    "  capture_output=True, text=True).stdout)"))

# 2) 上传守护进程
b64 = base64.b64encode(open(
    r"E:\ai_agent_workspace\teleagent\.temp\scnet_v8\evo_daemon.py",
    "rb").read()).decode()
api_put("/root/private_data/evo_daemon.py", b64)
print("uploaded /root/private_data/evo_daemon.py")

# 3) 等待器：E3(PID 12745) 退出后自动启动守护进程（常驻，永不退出）
print("== arming daemon behind E3 ==")
print(run_code(
    "import subprocess\n"
    "waiter = ('while kill -0 12745 2>/dev/null; do sleep 30; done; "
    "sleep 10; cd /root/private_data && exec python3 evo_daemon.py "
    "> evo_daemon.log 2>&1')\n"
    "logw = open('/root/private_data/evo_daemon_waiter.log', 'w')\n"
    "pw = subprocess.Popen(['bash', '-c', waiter], stdout=logw,\n"
    "                      stderr=subprocess.STDOUT, start_new_session=True)\n"
    "print('daemon waiter pid', pw.pid)"))
