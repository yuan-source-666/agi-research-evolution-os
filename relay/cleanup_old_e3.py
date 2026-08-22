#!/usr/bin/env python3
"""清理残留：杀旧 waiter 12912 + 旧 E3 12745（避免和新 E3 13078 抢 GPU）。"""
import re
src = open(r"E:\ai_agent_workspace\teleagent\.temp\scnet_v8\fix_and_relaunch_e3.py",
           encoding="utf-8").read()
head = src.split("# 1)")[0]  # 复用 BASE/TOKEN/api_put/run_code 定义
exec(head)

bash = ("kill 12912 2>/dev/null; sleep 2; kill 12745 2>/dev/null; sleep 3; "
        "echo ---after---; "
        "ps aux | grep -E 'phase_e3_arch|evo_daemon' | grep -v grep; "
        "echo ---gpu---; "
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader; "
        "echo ---end---")
code = ("import subprocess\n"
        "r = subprocess.run(['bash','-lc',%r],capture_output=True,text=True)\n"
        "print(r.stdout); print(r.stderr)\n" % bash)
print(run_code(code))
