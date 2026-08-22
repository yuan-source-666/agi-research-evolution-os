# -*- coding: utf-8 -*-
"""在远端内核执行 shell 命令的通用工具（复用 check_e3_status.py 的连接逻辑）"""
import sys

src = open("check_e3_status.py", encoding="utf-8").read()
exec(src.split("print(run_code(")[0])

cmd = sys.argv[1]
code = (
    "import subprocess\n"
    "r = subprocess.run(['bash','-lc', %r], capture_output=True, text=True)\n"
    "print(r.stdout)\n"
    "if r.stderr:\n"
    "    print('STDERR:', r.stderr[-800:])\n"
) % cmd
print(run_code(code))
