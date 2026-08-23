#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""远端状态采集：解析 scaled 实验日志/分片/GPU，输出单行 JSON（###JSON### 标记）。"""
import json, os, re, subprocess, glob

BASE = "/root/private_data/v8"
res = {"running": {}, "groups_done": [], "gpu": {}, "n_procs": 0}

# --- 运行进程数 ---
ps = subprocess.run(["bash", "-lc", "ps aux | grep exp_v8_scaled | grep -v grep"],
                    capture_output=True, text=True).stdout
res["n_procs"] = len([l for l in ps.splitlines() if l.strip()])

# --- 各 config 日志（log_<cfg>.txt，多组 append，取最后一行=当前组） ---
for f in sorted(glob.glob(BASE + "/scaled/log_*.txt")):
    cfg = os.path.basename(f)[4:-4]
    try:
        lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        continue
    last = ""
    for l in reversed(lines):
        if "步" in l:
            last = l
            break
    m = re.search(r"步 (\d+)/(\d+).*?loss=([\d.]+) val=([\d.]+) 阶段=(\S+) 神经=\[([\d, ]+)\] 分裂=(\d+) 修剪=(\d+)", last)
    if not m:
        continue
    entry = {"step": int(m.group(1)), "total": int(m.group(2)),
             "loss": float(m.group(3)), "val": float(m.group(4)), "phase": m.group(5),
             "neurons": [int(x) for x in m.group(6).split(",")],
             "splits": int(m.group(7)), "prunes": int(m.group(8))}
    # val 历史用于曲线（只取当前组：从最后一个含 "=====" 或组头之后的行不可靠，取尾部 300 行近似）
    hist = []
    for l in lines[-300:]:
        mm = re.search(r"步 (\d+)/(\d+).*?val=([\d.]+)", l)
        if mm:
            hist.append([int(mm.group(1)), float(mm.group(3))])
    entry["val_hist"] = hist[-200:]
    res["running"][cfg] = entry

# --- 已完成组（shard_*.json 增量落盘） ---
for f in sorted(glob.glob(BASE + "/scaled/shard_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if isinstance(d, dict):
        d = d.get("results") or d.get("groups") or []
    if isinstance(d, list):
        for g in d:
            if isinstance(g, dict):
                tf = g.get("tf") or {}
                res["groups_done"].append({
                    "cfg": g.get("cfg"), "seed": g.get("seed"),
                    "top1": tf.get("top1"), "top5": tf.get("top5"),
                    "val": g.get("best_val") if g.get("best_val") is not None else g.get("val"),
                    "neurons": g.get("active_neurons"),
                    "wall": g.get("wall_s")})

# --- GPU：torch 精确显存 + hy-smi 原文 ---
try:
    import torch
    free, total = torch.cuda.mem_get_info()
    res["gpu"]["free_gb"] = round(free / 2**30, 1)
    res["gpu"]["total_gb"] = round(total / 2**30, 1)
    res["gpu"]["used_gb"] = round((total - free) / 2**30, 1)
except Exception as e:
    res["gpu"]["err"] = str(e)[:120]
try:
    r = subprocess.run(["bash", "-lc", "hy-smi 2>&1 | head -40"],
                       capture_output=True, text=True).stdout
    res["gpu"]["raw"] = r[:1500]
    res["gpu"]["pcts"] = re.findall(r"(\d+)\s*%", r)[:12]
except Exception:
    pass

print("###JSON###" + json.dumps(res, ensure_ascii=True))
