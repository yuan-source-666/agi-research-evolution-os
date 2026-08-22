#!/usr/bin/env python3
"""确认 E3(13078) 与 waiter(13079) 进程存活状态"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "check_e3_status.py")).read().split("print(run_code(")[0])
print(run_code(
    "import subprocess\n"
    "print(subprocess.run(['bash','-lc',\n"
    "  'kill -0 13078 2>/dev/null && echo E3_ALIVE || echo E3_DEAD; "
    "kill -0 13079 2>/dev/null && echo WAITER_ALIVE || echo WAITER_DEAD; "
    "echo ---; ps aux | grep -E \"phase_e3|evo_daemon|relaunch\" | grep -v grep | awk "
    "\"{print $2, $11, $12}\"; echo ---end---'],\n"
    "  capture_output=True, text=True).stdout)"))
