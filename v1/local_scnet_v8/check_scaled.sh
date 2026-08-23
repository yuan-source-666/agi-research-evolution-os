#!/bin/bash
# 查看扩大实验进度：进程数 / 已保存分片 / GPU 聚合利用率
cd /root/private_data/v8
echo "=== procs ==="
ps aux | grep exp_v8_scaled | grep -v grep | awk '{print $2, $13, $14}'
echo "=== scaled dir ==="
ls -la scaled/ 2>/dev/null | grep -v '^total'
echo "=== DONE? ==="
ls scaled/DONE 2>/dev/null && cat scaled/DONE || echo "not yet"
echo "=== HCU x6 (2s间隔) ==="
for i in 1 2 3 4 5 6; do
  hy-smi | grep -E '^0 ' || hy-smi | sed -n '4p'
  sleep 2
done
echo "=== last lines of each shard (if any) ==="
for f in scaled/shard_*.json; do
  [ -f "$f" ] && echo "-- $f ($(wc -c < "$f") bytes)"
done
