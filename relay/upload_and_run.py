#!/usr/bin/env python3
"""上传 v8 实验包到 SCNet 实例并后台启动消融实验。
用法:
  HTTP_PROXY= HTTPS_PROXY= JUPYTER_TOKEN=<token> python upload_and_run.py [--check-only]
流程: mkdir -> 上传 4 个文件 -> Popen 后台启动 -> 立即返回
"""
import os
import sys
import json
import base64
import argparse
import requests

BASE = os.environ.get(
    "JUPYTER_BASE",
    "https://REDACTED_JUPYTER_BASE",
)
TOKEN = os.environ.get("JUPYTER_TOKEN", "REDACTED_JUPYTER_TOKEN")
LOCAL_DIR = r"E:\ai_agent_workspace\teleagent\.temp\scnet_v8"
REMOTE_DIR = "/root/private_data/v8"
FILES = ["bionic_llm_v7.py", "bionic_llm_v8.py", "corpus_zh_en.txt",
         "exp_v8_ablation_dcu.py"]

# 本地执行环境禁用代理（FlClash 会拦 [REDACTED-DOMAIN]）
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"


def api_put(path, content_b64):
    r = requests.put(
        BASE + "/api/contents" + path,
        params={"token": TOKEN},
        json={"content": content_b64, "format": "base64", "type": "file"},
        timeout=120)
    r.raise_for_status()
    return r.json()


def run_code(code, timeout=300):
    """在 Jupyter kernel 里执行（借用 skill 的 jupyter_exec 逻辑）"""
    import uuid
    import websocket
    r = requests.post(BASE + "/api/kernels", params={"token": TOKEN},
                      json={"name": "python3"}, timeout=30)
    r.raise_for_status()
    kid = r.json()["id"]
    ws_url = BASE.replace("https://", "wss://", 1) + \
        "/api/kernels/%s/channels?token=%s" % (kid, TOKEN)
    ws = websocket.create_connection(ws_url, timeout=timeout)
    msg_id = uuid.uuid4().hex
    req = {
        "header": {"msg_id": msg_id, "username": "agent", "session": msg_id,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "buffers": [],
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False,
                    "stop_on_error": True},
        "channel": "shell",
    }
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    # 1. 环境检查
    print("== 环境检查 ==")
    print(run_code(
        "import subprocess, torch\n"
        "print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())\n"
        "try:\n"
        "    print('dev', torch.cuda.get_device_name(0))\n"
        "except Exception as e:\n"
        "    print('no dev:', e)\n"
        "print(subprocess.run(['bash','-lc','ls /root/private_data 2>/dev/null | head; free -g | head -2'],"
        "capture_output=True,text=True).stdout)"))
    if args.check_only:
        return

    # 2. mkdir + 上传
    print("== 上传 ==")
    run_code("import os; os.makedirs('%s', exist_ok=True)" % REMOTE_DIR)
    for fn in FILES:
        p = os.path.join(LOCAL_DIR, fn)
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        api_put(REMOTE_DIR + "/" + fn, b64)
        print("  uploaded", fn)

    # 3. 后台启动（增量保存版脚本，中断不丢数据）
    print("== 启动 ==")
    code = (
        "import subprocess, os\n"
        "d = '%s'\n"
        "log = open(os.path.join(d, 'run.log'), 'w')\n"
        "p = subprocess.Popen(\n"
        "    ['python3', 'exp_v8_ablation_dcu.py',\n"
        "     '--configs', 'dev', 'fixed_small', 'fixed_large',\n"
        "     '--seeds', '0', '1', '2', '--steps', '800'],\n"
        "    cwd=d, stdout=log, stderr=subprocess.STDOUT,\n"
        "    start_new_session=True)\n"
        "print('pid', p.pid)\n" % REMOTE_DIR)
    print(run_code(code))
    print("完成。日志: %s/run.log，结果: %s/v8_ablation.json" % (REMOTE_DIR, REMOTE_DIR))


if __name__ == "__main__":
    main()
