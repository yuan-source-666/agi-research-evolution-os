#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""远端合并 scaled 实验 shard 并下载到本地 .temp/v8_scaled.json，打印统计。"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import upload_and_run as U

MERGE = '''import json, glob
d = []
for f in sorted(glob.glob('/root/private_data/v8/scaled/shard_*.json')):
    x = json.load(open(f))
    if isinstance(x, dict):
        x = x.get('results') or x.get('groups') or []
    d += x
print('###JSON###' + json.dumps({'groups': d}, ensure_ascii=True))
'''

# 上传 merge 脚本
import base64
U.api_put('/root/private_data/v8/merge_shards.py',
          base64.b64encode(MERGE.encode()).decode())

CODE = (
    "import subprocess\n"
    "r = subprocess.run(['bash','-lc','cd /root/private_data/v8 && python3 merge_shards.py'],"
    "capture_output=True,text=True)\n"
    "print(r.stdout)\n"
    "print('[RC]', r.returncode)\n"
    "if r.returncode != 0:\n    print(r.stderr[-400:])\n"
)
out = U.run_code(CODE, timeout=120)
m = re.search(r"###JSON###(\{.*\})", out, re.S)
if not m:
    print("FAILED:", out[-600:])
    sys.exit(1)
data = json.loads(m.group(1))
groups = data["groups"]
out_path = os.path.join(os.path.dirname(HERE), "v8_scaled.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(groups, fh, ensure_ascii=False, indent=1)
print("saved", out_path, "| groups:", len(groups))

# 统计
import statistics as st
by = {}
for g in groups:
    by.setdefault(g["cfg"], []).append(g)
for cfg in sorted(by):
    gs = by[cfg]
    t1 = [g["tf"]["top1"] for g in gs]
    vl = [g["best_val"] if g.get("best_val") is not None else g.get("val") for g in gs]
    neu = [g.get("active_neurons") for g in gs]
    print("%-12s n=%d top1=%.1f±%.1f  bestval=%s  neurons=%s  wall=%.0fs" % (
        cfg, len(gs), st.mean(t1), st.pstdev(t1),
        ["%.3f" % v for v in vl], neu, st.mean([g.get("wall_s", 0) for g in gs])))
