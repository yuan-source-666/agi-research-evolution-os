import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download
snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", local_dir="/root/private_data/DeepSeek-R1-Distill-Qwen-14B")
print("MODEL_DL_DONE", flush=True)
