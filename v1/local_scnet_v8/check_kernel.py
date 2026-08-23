# -*- coding: utf-8 -*-
"""查询对话内核的模型/adapter/历史状态（复用 chat_relay 的 kernel_exec）"""
import chat_relay as R

kid = open(R.KID_FILE).read().strip()
print("kernel:", kid, "alive:", R._kernel_alive(kid))
out, err = R.kernel_exec(
    "print('VER=%s ADAPTER=%s MODEL=%s HIST=%d NUMPARAM=%s' % ("
    "AGI_VER, ADAPTER_ON, MODEL_PATH, len(CHAT_HISTORY), "
    "sum(p.numel() for p in CHAT_MODEL.parameters())//10**6))", timeout=120, kid=kid)
print(out[-500:])
if err:
    print("ERR:", err[-300:])
