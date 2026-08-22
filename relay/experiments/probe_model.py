# -*- coding: utf-8 -*-
"""最小对照实验：不含任何脚手架，直接测模型对题干数字的保真度"""
import chat_relay as R

kid = open(R.KID_FILE).read().strip()

CODE = r'''
_msgs = [
    {"role": "system", "content": "你是一个严谨的解题助手。"},
    {"role": "user", "content": "一个球从10米高自由落下，每次弹起高度是前次的一半。"
     "请先原样复述题目给出的初始高度数字，然后计算总路程。"},
]
_txt = TOK.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True)
_enc = TOK(_txt, return_tensors="pt").to(CHAT_MODEL.device)
_out = CHAT_MODEL.generate(**_enc, max_new_tokens=300, do_sample=True,
                            temperature=0.6, top_p=0.9)
print("__RAW_START__")
print(TOK.decode(_out[0][_enc["input_ids"].shape[1]:], skip_special_tokens=True))
print("__RAW_END__")
'''
out, err = R.kernel_exec(CODE, timeout=300, kid=kid)
if "__RAW_START__" in out:
    print(out.split("__RAW_START__", 1)[1].split("__RAW_END__", 1)[0].strip())
else:
    print("ERR:", (err or "")[-500:], "| raw:", out[-300:])
