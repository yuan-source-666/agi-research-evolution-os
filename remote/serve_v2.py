"""
Qwen3.8-27B-Uncensored-FP8 推理服务 v2
- transformers 5.8.1 (支持 qwen3_5 架构)
- OpenAI 兼容 API + 内置网页聊天界面 /chat
- 监听 6666 端口
"""
import torch, json, time, uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

MODEL_PATH = "/root/private_data/Qwen3.8-27B-Uncensored-FP8"
PORT = 6666
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60, flush=True)
print("  Qwen3.8-27B-Uncensored-FP8 推理服务 v2", flush=True)
print(f"  设备: {DEVICE}  端口: {PORT}  transformers: {__import__('transformers').__version__}", flush=True)
print("=" * 60, flush=True)

print("[1/2] 加载 Tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("  Tokenizer OK", flush=True)

print("[2/2] 加载模型 (FP8 反量化为 bf16)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=FineGrainedFP8Config(dequantize=True),
    dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()
print("  模型加载完成", flush=True)
if DEVICE == "cuda":
    vram = torch.cuda.memory_allocated(0) / 1024**3
    free, total = torch.cuda.mem_get_info(0)
    print(f"  显存占用: {vram:.1f} GB / 总 {total/1024**3:.1f} GB", flush=True)

app = FastAPI(title="Qwen3.8-27B API", version="2.0")


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


def _generate(messages, temperature, top_p, max_tokens):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    gen_kwargs = {
        "max_new_tokens": max_tokens or 2048,
        "temperature": max(temperature or 0.7, 0.01),
        "top_p": top_p or 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True)
    return response_text, input_len, len(generated)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        response_text, input_len, gen_len = _generate(messages, request.temperature, request.top_p, request.max_tokens)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen3.8-27b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": input_len, "completion_tokens": gen_len, "total_tokens": input_len + gen_len},
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
            "usage": {"prompt_tokens": input_len, "completion_tokens": len(generated), "total_tokens": input_len + len(generated)},
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "service": "Qwen3.8-27B-Uncensored-FP8 API v2",
        "chat_ui": "/chat",
        "endpoints": {"chat": "/v1/chat/completions", "completion": "/v1/completions", "models": "/v1/models", "health": "/health"},
    }


CHAT_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen3.8-27B 聊天</title>
<style>
  :root { --bg:#0f1220; --panel:#171a2b; --card:#1e2237; --border:#2a2f4a; --text:#e6e8f0; --muted:#8b90a8; --accent:#4c7dff; --accent2:#6d5cff; --user:#2563eb; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:radial-gradient(1200px 600px at 50% -10%,#1b2140 0%,var(--bg) 60%); color:var(--text); height:100vh; display:flex; flex-direction:column; }
  header { padding:16px 22px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:12px; background:rgba(23,26,43,.7); backdrop-filter:blur(8px); }
  header .dot { width:10px; height:10px; border-radius:50%; background:#34d399; box-shadow:0 0 10px #34d399; }
  header h1 { font-size:16px; font-weight:600; }
  header .sub { font-size:12px; color:var(--muted); }
  #chat { flex:1; overflow-y:auto; padding:24px; display:flex; flex-direction:column; gap:16px; max-width:860px; width:100%; margin:0 auto; }
  .msg { display:flex; gap:10px; max-width:85%; }
  .msg.user { align-self:flex-end; flex-direction:row-reverse; }
  .msg .avatar { width:34px; height:34px; border-radius:9px; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-size:15px; font-weight:700; }
  .msg.assistant .avatar { background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#fff; }
  .msg.user .avatar { background:#2563eb; color:#fff; }
  .bubble { padding:12px 15px; border-radius:14px; background:var(--card); border:1px solid var(--border); line-height:1.65; font-size:15px; white-space:pre-wrap; word-break:break-word; }
  .msg.user .bubble { background:var(--user); border-color:transparent; border-radius:14px 14px 4px 14px; }
  .msg.assistant .bubble { border-radius:14px 14px 14px 4px; }
  .bubble.loading::after { content:"…"; animation:blink 1s infinite; }
  @keyframes blink { 50% { opacity:.2; } }
  footer { padding:16px 22px; border-top:1px solid var(--border); background:rgba(23,26,43,.7); }
  .inputbar { max-width:860px; margin:0 auto; display:flex; gap:10px; }
  textarea { flex:1; resize:none; background:var(--card); border:1px solid var(--border); border-radius:12px; color:var(--text); padding:12px 14px; font-size:15px; font-family:inherit; line-height:1.5; height:52px; max-height:160px; outline:none; }
  textarea:focus { border-color:var(--accent); }
  button { background:linear-gradient(135deg,var(--accent),var(--accent2)); border:none; color:#fff; padding:0 22px; border-radius:12px; font-size:15px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .hint { text-align:center; color:var(--muted); font-size:12px; margin-top:8px; }
</style>
</head>
<body>
<header>
  <span class="dot"></span>
  <div><h1>Qwen3.8-27B-Uncensored-FP8</h1><div class="sub">海光 DCU 本地推理 · FP8 量化</div></div>
</header>
<div id="chat"></div>
<footer>
  <div class="inputbar">
    <textarea id="input" placeholder="输入消息，回车发送，Shift+回车换行"></textarea>
    <button id="send">发送</button>
  </div>
  <div class="hint">回复速度取决于模型推理，首次可能较慢</div>
</footer>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let messages = [];
let busy = false;

function addMsg(role, text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '我' : 'Q';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;
  addMsg('user', text);
  messages.push({ role: 'user', content: text });
  input.value = '';
  busy = true;
  sendBtn.disabled = true;
  const loading = addMsg('assistant', '');
  loading.classList.add('loading');
  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'qwen3.8-27b', messages, temperature: 0.7, top_p: 0.9, max_tokens: 2048 })
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    const reply = data.choices[0].message.content;
    loading.classList.remove('loading');
    loading.textContent = reply;
    messages.push({ role: 'assistant', content: reply });
  } catch (e) {
    loading.classList.remove('loading');
    loading.textContent = '⚠️ 请求失败: ' + e.message;
  }
  busy = false;
  sendBtn.disabled = false;
  input.focus();
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
addMsg('assistant', '你好！我是 Qwen3.8-27B，部署在海光 DCU 上。随便问点什么吧。');
</script>
</body>
</html>"""


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return CHAT_HTML


if __name__ == "__main__":
    import uvicorn
    print(f"服务启动，监听 0.0.0.0:{PORT}", flush=True)
    print(f"网页聊天: http://<host>:{PORT}/chat", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
