#!/bin/bash
# v8 扩大预算实验编排：共享卡空闲显存 ~5GB，每进程 ~1.4GB
# -> 最多 3 个训练进程并行，第 4 个配置等最先进程结束后补位
cd /root/private_data/v8
mkdir -p scaled
rm -f scaled/DONE

echo "[start] $(date) 空闲显存检查:"
python3 -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'  free {f/2**30:.2f} GB / {t/2**30:.2f} GB')"

run_cfg () {
  local cfg=$1
  python3 exp_v8_scaled_dcu.py --configs "$cfg" --seeds 0 1 2 --steps 3200 \
    --out "scaled/shard_$cfg.json" > "scaled/log_$cfg.txt" 2>&1
  echo "[done] $cfg at $(date)"
}

run_cfg dev &
run_cfg fixed_small &
run_cfg fixed_mid &

# 等任意一个结束，再启动第 4 个（保证任意时刻 <= 3 进程）
wait -n
run_cfg fixed_large &

wait
echo "ALL_DONE $(date)" > scaled/DONE
echo "ALL_DONE"
