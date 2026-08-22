#!/usr/bin/env python3
"""杀旧 daemon(15040)、启动类脑架构 v2 daemon。档案持久 → 冠军无缝继承。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "check_e3_status.py")).read().split("print(run_code(")[0])

print(run_code(
    "import subprocess\n"
    "print(subprocess.run(['bash','-lc',\n"
    "  'kill 15040 2>/dev/null; sleep 5; "
    "kill -0 15040 2>/dev/null && echo OLD_STILL_ALIVE || echo OLD_KILLED; "
    "nvidia-smi --query-gpu=memory.used --format=csv,noheader'],\n"
    "  capture_output=True, text=True).stdout)"))

print("== starting brain-arch v2 daemon ==")
print(run_code(
    "import subprocess\n"
    "logd = open('/root/private_data/evo_daemon.log', 'w')\n"
    "p = subprocess.Popen(['bash', '-c',\n"
    "  'cd /root/private_data && exec python3 evo_daemon.py'],\n"
    "  stdout=logd, stderr=subprocess.STDOUT, start_new_session=True)\n"
    "print('daemon pid', p.pid)"))
