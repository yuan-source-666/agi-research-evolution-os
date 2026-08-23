#!/usr/bin/env python3
"""杀旧 daemon(18879 brain-v2)、启动 v3 daemon（参数自膨胀）。档案持久 → 冠军无缝继承。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "check_e3_status.py")).read().split("print(run_code(")[0])

print(run_code(
    "import subprocess\n"
    "print(subprocess.run(['bash','-lc',\n"
    "  'kill 18879 2>/dev/null; sleep 5; "
    "kill -0 18879 2>/dev/null && echo OLD_STILL_ALIVE || echo OLD_KILLED; "
    "ps aux | grep evo_daemon | grep -v grep; "
    "nvidia-smi --query-gpu=memory.used --format=csv,noheader'],\n"
    "  capture_output=True, text=True).stdout)"))

print("== starting v3 daemon (parameter self-growth) ==")
print(run_code(
    "import subprocess\n"
    "logd = open('/root/private_data/evo_daemon.log', 'a')\n"
    "logd.write('\\n===== v3 restart %s =====\\n' % __import__('time').strftime('%F %T'))\n"
    "p = subprocess.Popen(['bash', '-c',\n"
    "  'cd /root/private_data && exec python3 evo_daemon.py'],\n"
    "  stdout=logd, stderr=subprocess.STDOUT, start_new_session=True)\n"
    "print('daemon pid', p.pid)"))
