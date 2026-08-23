"""
Qwen3.8-27B-Uncensored-FP8 推理服务
监听 6666 端口，提供 OpenAI 兼容的 API 接口
"""
import torch
import json
import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/private_data/Qwen3.8-27B-Uncensored-FP8"
PORT = 6666
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{'='*60}")
print(f"  Qwen3.8-27B-Uncensored-FP8 推理服务")
print(f"  设备: {DEVICE}")
print(f"  端口: {PORT}")
print(f"{'='*60}")

print(f"\n[1/2] 正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"  Tokenizer 加载完成")

print(f"\n[2/2] 正在加载模型到 {DEVICE}...")
print(f"  （27B FP8 模型，约需 1-3 分钟，请耐心等待...）")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True,
)
model.eval()
print(f"  模型加载完成！")

if DEVICE == "cuda":
    vram_used = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  显存占用: {vram_used:.1f} GB")

print(f"\n{'='*60}")
print(f"  服务启动中... 监听端口 {PORT}")
print(f"{'='*60}\n")

app = FastAPI(title="Qwen3.8-27B API", version="1.0")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen3.8-27b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False

class GenerateRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048

@app.get("/health")
@app.get("/v1/health")
async def health():
    return {"status": "ok", "model": "Qwen3.8-27B-Uncensored-FP8"}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "qwen3.8-27b", "object": "model", "created": int(time.time()), "owned_by": "local"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        gen_kwargs = {
            "max_new_tokens": request.max_tokens or 2048,
            "temperature": max(request.temperature or 0.7, 0.01),
            "top_p": request.top_p or 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response_text = tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen3.8-27b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": input_len, "completion_tokens": len(generated), "total_tokens": input_len + len(generated)}
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: GenerateRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
        gen_kwargs = {
            "max_new_tokens": request.max_tokens or 2048,
            "temperature": max(request.temperature or 0.7, 0.01),
            "top_p": request.top_p or 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response_text = tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "qwen3.8-27b",
            "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": input_len, "completion_tokens": len(generated), "total_tokens": input_len + len(generated)}
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "Qwen3.8-27B-Uncensored-FP8 API",
        "endpoints": {"chat": "/v1/chat/completions", "completion": "/v1/completions", "models": "/v1/models", "health": "/health"},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
