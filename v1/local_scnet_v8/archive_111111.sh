#!/bin/bash
# AGI 成就归档脚本 v1 — 2026-08-22
# 目标: /public/home/[REDACTED-CLUSTER-USER]/111111
set -u
DEST=/public/home/[REDACTED-CLUSTER-USER]/111111
SRC=/root/private_data
mkdir -p "$DEST/remote_full"

echo "[1/4] rsync 成就文件..."
rsync -a --stats \
  --exclude='/111111' \
  --exclude='/models' \
  --exclude='/SothisAI' \
  --exclude='/.SothisAI' \
  --exclude='/Qwen2.5-1.5B-Instruct' \
  --exclude='/Qwen2.5-7B-Instruct' \
  --exclude='/Qwen2.5-32B-Instruct-GPTQ-Int4' \
  --exclude='/DeepSeek-R1-Distill-Qwen-14B' \
  --exclude='/.cache' \
  --exclude='/.pnpm-store' \
  --exclude='/modelscope_cache' \
  --exclude='/.Trash-0' \
  --exclude='/.ssh' \
  --exclude='/.sc' \
  --exclude='/.kube' \
  --exclude='/.mozilla' \
  --exclude='/.jupyterLab' \
  --exclude='/.local' \
  --exclude='/.config' \
  --exclude='/.ipynb_checkpoints' \
  --exclude='__pycache__' \
  "$SRC/" "$DEST/remote_full/" > "$DEST/rsync.log" 2>&1
echo "rsync exit=$?"

echo "[2/4] 生成 sha256 校验清单(唯一性复用)..."
cd "$DEST"
find remote_full local_scnet_v8 -type f ! -name 'SHA256SUMS.txt' -print0 2>/dev/null \
  | xargs -0 sha256sum > SHA256SUMS.txt 2>>"$DEST/rsync.log"
echo "sha256 entries: $(wc -l < SHA256SUMS.txt)"

echo "[3/4] 统计..."
echo "total files: $(find remote_full local_scnet_v8 -type f | wc -l)" > ARCHIVE_STATS.txt
echo "total size: $(du -sh remote_full local_scnet_v8 | tail -1)" >> ARCHIVE_STATS.txt
du -sh remote_full local_scnet_v8 >> ARCHIVE_STATS.txt 2>/dev/null

echo "[4/4] done"
cat ARCHIVE_STATS.txt
