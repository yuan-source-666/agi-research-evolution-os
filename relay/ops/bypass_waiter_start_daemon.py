#!/usr/bin/env python3
"""waiter PID 复用卡死 → 直接杀 waiter、手动启动自适应版 daemon（GPU 已空闲）"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "check_e3_status.py")).read().split("print(run_code(")[0])

# 1) 看 PID 13078 现在是谁（验证 PID 复用）
print(run_code(
    "import subprocess\n"
    "print(subprocess.run(['bash','-lc',\n"
    "  'ps -p 13078 -o pid,cmd --no-headers; echo ---; "
    "kill 13079 2>/dev/null; sleep 2; "
    "kill -0 13079 2>/dev/null && echo WAITER_STILL_ALIVE || echo WAITER_KILLED'],\n"
    "  capture_output=True, text=True).stdout)"))

# 2) 直接启动自适应版 daemon
print("== starting adaptive daemon ==")
print(run_code(
    "import subprocess\n"
    "logd = open('/root/private_data/evo_daemon.log', 'w')\n"
    "p = subprocess.Popen(['bash', '-c',\n"
    "  'cd /root/private_data && exec python3 evo_daemon.py'],\n"
    "  stdout=logd, stderr=subprocess.STDOUT, start_new_session=True)\n"
    "print('daemon pid', p.pid)"))
