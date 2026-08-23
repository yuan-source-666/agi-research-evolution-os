
import os, sys, time
os.environ["MODELSCOPE_CACHE"] = "/root/private_data"

from modelscope import snapshot_download
print("Starting download Qwen/Qwen2.5-32B-Instruct...", flush=True)
t0 = time.time()
model_dir = snapshot_download("Qwen/Qwen2.5-32B-Instruct", cache_dir="/root/private_data")
print(f"Downloaded to: {model_dir}", flush=True)
print(f"Time: {time.time()-t0:.1f}s", flush=True)

# Write path to file
with open("/root/private_data/.bf16_model_path", "w") as f:
    f.write(model_dir)
print("DOWNLOAD_DONE", flush=True)
