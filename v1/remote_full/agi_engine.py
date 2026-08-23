#!/usr/bin/env python3
"""AGI Inference Engine - Qwen2.5-32B-Instruct BF16 on AMD DCU.
OpenAI-compatible API server with function calling support."""

import os, sys, json, time, uuid, traceback, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn

# Try BF16 model path, fallback to GPTQ if BF16 not available
_BF16_PATH = "/root/private_data/models/Qwen--Qwen2.5-32B-Instruct/snapshots/master"
_GPTQ_PATH = "/root/private_data/Qwen2.5-32B-Instruct-GPTQ-Int4"
import os as _os
if _os.path.exists(_os.path.join(_BF16_PATH, "config.json")):
    MODEL_PATH = _BF16_PATH
    MODEL_DTYPE = "bf16"
else:
    MODEL_PATH = _GPTQ_PATH
    MODEL_DTYPE = "gptq"
PORT = 8080

# ============================================================
# Model Loading
# ============================================================
print("=" * 60, flush=True)
print(f"[AGI Engine] Initializing Qwen2.5-32B-Instruct ({MODEL_DTYPE.upper()})", flush=True)
print(f"[AGI Engine] Model path: {MODEL_PATH}", flush=True)
print(f"[AGI Engine] GPU: {torch.cuda.get_device_name(0)}", flush=True)
vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"[AGI Engine] VRAM: {vram_total:.1f} GB", flush=True)
print("=" * 60, flush=True)

print("[1/4] Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"  Tokenizer loaded. Vocab size: {len(tokenizer)}", flush=True)

print(f"[2/4] Loading model ({MODEL_DTYPE})...", flush=True)
t0 = time.time()
_load_kwargs = dict(device_map="auto", trust_remote_code=True, attn_implementation="eager")
if MODEL_DTYPE == "bf16":
    _load_kwargs["dtype"] = torch.bfloat16
else:
    _load_kwargs["dtype"] = torch.float16
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **_load_kwargs)
model = model.eval()
load_time = time.time() - t0
print(f"  Model loaded in {load_time:.1f}s", flush=True)
print(f"  VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB", flush=True)
print(f"  Model dtype: {next(model.parameters()).dtype}", flush=True)

print("[3/4] Warmup...", flush=True)
t0 = time.time()
wi = tokenizer("Hello", return_tensors="pt").to(model.device)
with torch.no_grad():
    _ = model.generate(**wi, max_new_tokens=5, do_sample=False)
print(f"  Warmup done in {time.time() - t0:.1f}s", flush=True)

print("[4/4] Building API...", flush=True)
print("=" * 60, flush=True)

# ============================================================
# Pydantic Models
# ============================================================
class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str = ""
    type: str = "function"
    function: ToolCallFunction

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class ToolFunction(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = {}

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class ChatRequest(BaseModel):
    model: str = "qwen2.5-32b"
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None
    stop: Optional[List[str]] = None

class CompletionRequest(BaseModel):
    model: str = "qwen2.5-32b"
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

# ============================================================
# Tool Call Parsing
# ============================================================
TOOL_CALL_START = "[TOOL_CALL]"
TOOL_CALL_END = "[/TOOL_CALL]"

def build_tool_system_prompt(tools):
    tool_descs = []
    for t in tools:
        tf = t.function
        params_str = json.dumps(tf.parameters, ensure_ascii=False, indent=2)
        tool_descs.append(f"- {tf.name}: {tf.description}\n  Parameters: {params_str}")
    tools_str = "\n".join(tool_descs)
    return (
        "You have access to the following tools:\n\n"
        f"{tools_str}\n\n"
        "When you need to use a tool, output a tool call in this exact format:\n"
        f"{TOOL_CALL_START}" + "{\"name\": \"tool_name\", \"arguments\": {...}}" + f"{TOOL_CALL_END}\n\n"
        "You can call multiple tools by emitting multiple tool call blocks.\n"
        "After tool results are provided, continue your response naturally.\n"
        "If you don't need a tool, just respond normally."
    )

def parse_tool_calls(text):
    pattern = re.compile(
        re.escape(TOOL_CALL_START) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_END),
        re.DOTALL
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return text.strip(), None

    content_before = text[:matches[0].start()].strip()
    tool_calls = []
    for i, m in enumerate(matches):
        raw = m.group(1).strip()
        try:
            parsed = json.loads(raw)
            tc = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": parsed.get("name", ""),
                    "arguments": json.dumps(parsed.get("arguments", {}), ensure_ascii=False)
                }
            }
            tool_calls.append(tc)
        except json.JSONDecodeError:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": "unknown", "arguments": raw}
            })

    content_after = text[matches[-1].end():].strip()
    final_content = content_before
    if content_after:
        final_content = f"{final_content}\n{content_after}" if final_content else content_after
    return final_content, tool_calls

# ============================================================
# Generation Helpers
# ============================================================
def messages_to_prompt(messages, tools=None):
    system_parts = []
    for msg in messages:
        if msg.role == "system" and msg.content:
            system_parts.append(msg.content)
    if tools:
        system_parts.append(build_tool_system_prompt(tools))

    prompt_parts = []
    if system_parts:
        prompt_parts.append("<|im_start|>system\n" + "\n\n".join(system_parts) + "<|im_end|>")

    for msg in messages:
        if msg.role == "system":
            continue
        role = msg.role
        content = msg.content or ""
        if msg.tool_calls:
            tc_parts = []
            for tc in msg.tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args_dict = json.loads(args)
                    except Exception:
                        args_dict = {"raw": args}
                else:
                    args_dict = args
                tc_str = json.dumps({"name": name, "arguments": args_dict}, ensure_ascii=False)
                tc_parts.append(f"{TOOL_CALL_START}{tc_str}{TOOL_CALL_END}")
            content = (content + "\n" + "\n".join(tc_parts)) if content else "\n".join(tc_parts)

        if msg.role == "tool":
            tool_name = msg.name or "tool"
            prompt_parts.append(f"<|im_start|>user\n[Tool Result: {tool_name}]\n{content}<|im_end|>")
        else:
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)

def generate_response(messages, max_tokens=1024, temperature=0.7, top_p=0.9, tools=None, stop=None):
    prompt = messages_to_prompt(messages, tools)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.01,
        "temperature": max(temperature, 0.01),
        "top_p": top_p,
        "eos_token_id": eos_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    text = text.replace("<|im_end|>", "").strip()
    return text

def generate_stream(messages, max_tokens=1024, temperature=0.7, top_p=0.9, tools=None, stop=None):
    prompt = messages_to_prompt(messages, tools)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.01,
        "temperature": max(temperature, 0.01),
        "top_p": top_p,
        "eos_token_id": eos_id,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    thread = Thread(target=lambda: model.generate(**inputs, **gen_kwargs))
    thread.start()
    buffer = ""
    for text in streamer:
        buffer += text
        if "<|im_end|>" in buffer:
            buffer = buffer.replace("<|im_end|>", "")
            break
        yield text
    thread.join()

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="AGI Inference Engine", version="1.0.0")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": f"Qwen2.5-32B-Instruct-{MODEL_DTYPE.upper()}",
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        "vram_total_gb": round(vram_total, 2),
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "qwen2.5-32b",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "agi-engine",
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        if req.stream:
            return StreamingResponse(stream_chat(req), media_type="text/event-stream")
        t0 = time.time()
        raw_text = generate_response(
            messages=req.messages, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p,
            tools=req.tools, stop=req.stop,
        )
        gen_time = time.time() - t0
        content, tool_calls = parse_tool_calls(raw_text)
        message = {"role": "assistant", "content": content if content else None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = "tool_calls" if tool_calls else "stop"
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
            "timings": {"generation_time_s": round(gen_time, 3)},
        }
        return JSONResponse(content=response)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def stream_chat(req: ChatRequest):
    try:
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        for token in generate_stream(
            messages=req.messages, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p,
            tools=req.tools, stop=req.stop,
        ):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        final = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    try:
        t0 = time.time()
        inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        gen_kwargs = {
            "max_new_tokens": req.max_tokens,
            "do_sample": req.temperature > 0.01,
            "temperature": max(req.temperature, 0.01),
            "top_p": req.top_p,
            "pad_token_id": tokenizer.eos_token_id,
        }
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0, input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        gen_time = time.time() - t0
        response = {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": input_len, "completion_tokens": len(new_tokens), "total_tokens": input_len + len(new_tokens)},
            "timings": {"generation_time_s": round(gen_time, 3)},
        }
        return JSONResponse(content=response)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Main Entry Point
# ============================================================
if __name__ == "__main__":
    print(f"[AGI Engine] Starting server on 0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")