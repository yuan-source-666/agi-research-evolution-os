#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地轮询器：每 20s 从 SCNet 拉取 v8_status.py 输出，写 monitor/status.json。"""
import json, os, re, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import upload_and_run as U

OUT = os.path.join(HERE, "monitor", "status.json")
INTERVAL = 20

CODE = (
    "import subprocess\n"
    "r = subprocess.run(['bash','-lc','cd /root/private_data/v8 && python3 v8_status.py'],"
    "capture_output=True,text=True)\n"
    "print(r.stdout)\n"
    "print('[RC]', r.returncode)\n"
    "if r.returncode != 0:\n    print(r.stderr[-400:])\n"
)


def poll_once():
    out = U.run_code(CODE, timeout=150)
    m = re.search(r"###JSON###(\{.*\})", out, re.S)
    if not m:
        return {"ok": False, "error": (out or "")[-800:], "ts": time.time()}
    data = json.loads(m.group(1))
    data["ok"] = True
    data["ts"] = time.time()
    return data


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("poller started, interval", INTERVAL, "s", flush=True)
    while True:
        try:
            data = poll_once()
        except Exception:
            data = {"ok": False, "error": traceback.format_exc()[-800:], "ts": time.time()}
        tmp = OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, OUT)
        print(time.strftime("%H:%M:%S"), "ok" if data.get("ok") else "ERR",
              "procs=%s done=%s" % (data.get("n_procs"), len(data.get("groups_done") or [])), flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
