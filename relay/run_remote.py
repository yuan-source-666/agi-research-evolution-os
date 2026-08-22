#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用远程驱动：上传文件到 SCNet + 远程执行 bash 命令。
用法:
  JUPYTER_TOKEN=<token> python run_remote.py --upload a.py b.py --exec "cd ... && python3 a.py"
依赖同目录 upload_and_run.py 的 api_put / run_code。
"""
import sys
import base64
import argparse
import upload_and_run as U

REMOTE_DIR = "/root/private_data/v8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", nargs="+", default=[],
                    help="本地文件路径，上传到 REMOTE_DIR")
    ap.add_argument("--exec", default=None,
                    help="远程 bash 命令（阻塞执行，返回尾部输出）")
    ap.add_argument("--tail", type=int, default=2500,
                    help="远程输出保留的尾部字符数")
    ap.add_argument("--timeout", type=int, default=600,
                    help="kernel websocket 超时秒数")
    a = ap.parse_args()

    for local in a.upload:
        remote = REMOTE_DIR + "/" + local.replace("\\", "/").split("/")[-1]
        b64 = base64.b64encode(open(local, "rb").read()).decode()
        U.api_put(remote, b64)
        print("uploaded", local, "->", remote)

    if a.exec:
        code = (
            "import subprocess\n"
            "r = subprocess.run(['bash','-lc', %r], capture_output=True, text=True)\n"
            "print(r.stdout[-%d:])\n"
            "if r.returncode != 0:\n"
            "    print('[RC]', r.returncode)\n"
            "    print(r.stderr[-1500:])\n" % (a.exec, a.tail))
        print(U.run_code(code, timeout=a.timeout))


if __name__ == "__main__":
    main()
