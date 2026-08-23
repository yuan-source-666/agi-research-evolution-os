# -*- coding: utf-8 -*-
"""下载 Qwen2.5-1.5B-Instruct（提案生成器，约 3GB，适配 5.5GB 空闲显存）。"""
import os
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

DEST = "/root/private_data/Qwen2.5-1.5B-Instruct"
t0 = time.time()
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir=DEST)
print("MODEL_DL_DONE %.1fs -> %s" % (time.time() - t0, DEST), flush=True)
