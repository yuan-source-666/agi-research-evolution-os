#!/usr/bin/env python3
"""E3 下载阻塞最小修复：
1. 先杀 daemon waiter(12912) 防止杀 E3 后 daemon 抢跑
2. 杀旧 E3(12745) + 清半成品数据
3. 上传本地多镜像修复版 phase_e3_arch.py（上次重启传的还是旧版，这是阻塞根因）
4. 重启 E3 并重新挂 daemon waiter（evo_daemon.py 已是 P0 修复版）
"""
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


# 1) 先杀 waiter，再杀 E3，清半成品
print("== cleanup (waiter first!) ==")
print(run_code(
    "import subprocess\n"
    "print(subprocess.run(['bash','-lc',\n"
    "  'kill 12912 2>/dev/null; sleep 2; "
    "pkill -f evo_daemon 2>/dev/null; "
    "pkill -f phase_e3_arch 2>/dev/null; "
    "rm -f /root/private_data/ag_news_train.csv*; sleep 3; "
    "echo ---procs---; "
    "ps aux | grep -E \"phase_e3|evo_daemon|kill -0\" | grep -v grep; "
    "echo ---done---'],\n"
    "  capture_output=True, text=True).stdout)"))

# 2) 上传多镜像修复版 E3
b64 = base64.b64encode(open(
    r"E:\ai_agent_workspace\teleagent\.temp\scnet_v8\phase_e3_arch.py",
    "rb").read()).decode()
api_put("/root/private_data/phase_e3_arch.py", b64)
print("uploaded /root/private_data/phase_e3_arch.py (multi-mirror version)")

# 校验远端确实是多镜像版
r = requests.get(BASE + "/api/contents/root/private_data/phase_e3_arch.py",
                 params={"token": TOKEN, "format": "base64"}, timeout=60)
raw = base64.b64decode(r.json()["content"]).decode()
print("remote jsdelivr check:", "OK" if "cdn.jsdelivr.net" in raw else "MISSING!")

# 3) 重启 E3 + 挂 daemon waiter（P0 修复版）
print("== relaunching E3-ARCH + arming daemon ==")
print(run_code(
    "import subprocess\n"
    "log3 = open('/root/private_data/phase_e3_arch.log', 'w')\n"
    "p3 = subprocess.Popen(['bash', '-c',\n"
    "  'cd /root/private_data && exec python3 phase_e3_arch.py'],\n"
    "  stdout=log3, stderr=subprocess.STDOUT, start_new_session=True)\n"
    "print('e3_arch pid', p3.pid)\n"
    "waiter = ('while kill -0 %d 2>/dev/null; do sleep 30; done; "
    "sleep 10; cd /root/private_data && exec python3 evo_daemon.py "
    "> evo_daemon.log 2>&1') % p3.pid\n"
    "logw = open('/root/private_data/evo_daemon_waiter.log', 'w')\n"
    "pw = subprocess.Popen(['bash', '-c', waiter], stdout=logw,\n"
    "                      stderr=subprocess.STDOUT, start_new_session=True)\n"
    "print('daemon waiter pid', pw.pid)"))
