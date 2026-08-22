# -*- coding: utf-8 -*-
"""模型本体体检：tokenizer / 简单复述 / 简单算术 / config"""
import chat_relay as R

kid = open(R.KID_FILE).read().strip()

CODE = r'''
import json
print("__T1__", TOK("10").input_ids, TOK("11").input_ids)
_cfg = json.load(open(MODEL_PATH + "/config.json"))
print("__T2__ arch=%s layers=%s hidden=%s vocab=%s" % (
    _cfg.get("architectures"), _cfg.get("num_hidden_layers"),
    _cfg.get("hidden_size"), _cfg.get("vocab_size")))
import os
_sh = sorted(f for f in os.listdir(MODEL_PATH) if f.endswith(".safetensors"))
print("__T3__ shards=%d total=%.2fGB" % (
    len(_sh), sum(os.path.getsize(MODEL_PATH + "/" + f) for f in _sh) / 1e9))

def _qa(q, mx=80):
    _m = [{"role": "user", "content": q}]
    _t = TOK.apply_chat_template(_m, tokenize=False, add_generation_prompt=True)
    _e = TOK(_t, return_tensors="pt").to(CHAT_MODEL.device)
    _o = CHAT_MODEL.generate(**_e, max_new_tokens=mx, do_sample=False)
    return TOK.decode(_o[0][_e["input_ids"].shape[1]:], skip_special_tokens=True).strip()

print("__T4__", repr(_qa("请原样输出这个数字：10")))
print("__T5__", repr(_qa("3加5等于几？只回答数字。")))
print("__T6__", repr(_qa("从10米高的地方扔下一个球，球第一次落地时下落了多少米？只回答数字。")))
print("__T7__", repr(_qa("一个数列首项是10，公比是1/2，无穷项和的2倍加上首项是多少？")))
'''
out, err = R.kernel_exec(CODE, timeout=600, kid=kid)
print(out[-1500:] if out else "")
if err:
    print("ERR:", err[-300:])
