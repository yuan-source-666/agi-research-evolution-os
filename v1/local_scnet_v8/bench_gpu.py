#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SCNet DCU 性能基准：
  1. torch.cuda.mem_get_info 查真实空闲/总显存（hy-smi 显示 91% 占用之谜）
  2. 单进程训练时采样 HCU% 利用率
  3. 对比 在线分词 vs 预分词缓存 的每步耗时
"""
import os, sys, time, threading, subprocess, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch

f, t = torch.cuda.mem_get_info()
print(f"[mem] free {f/2**30:.2f} GB / total {t/2**30:.2f} GB", flush=True)

from bionic_llm_v8 import BionicLLMv8, DevelopmentEngine
from bionic_llm_v7 import BPETokenizer

DEVICE = 'cuda'

_orig_seq_for = DevelopmentEngine._seq_for

def _seq_for_dev(self, texts, n):
    X, Y = _orig_seq_for(self, texts, n)
    return X.to(DEVICE), Y.to(DEVICE)

tok = BPETokenizer()
with open(os.path.join(HERE, 'corpus_zh_en.txt'), encoding='utf-8') as fp:
    texts = [ln.strip() for ln in fp if ln.strip()]
tok.fit(texts)
rng = random.Random(0)
rng.shuffle(texts)
split = int(len(texts) * 0.85)
tr, va = texts[:split], texts[split:]
print(f"[corpus] {len(texts)} lines, vocab {len(tok.stoi)}", flush=True)

# ---- hy-smi 采样线程 ----
samples = []
_stop = threading.Event()

def sampler():
    while not _stop.is_set():
        try:
            out = subprocess.run(['hy-smi'], capture_output=True,
                                 text=True, timeout=10).stdout
            for line in out.splitlines():
                ps = line.split()
                # HCU  Temp  AvgPwr  Perf  PwrCap  VRAM%  HCU% ...
                if len(ps) >= 8 and ps[0] == '0' and ps[1].endswith('C'):
                    samples.append((ps[5], ps[6]))
        except Exception:
            pass
        time.sleep(1.0)


def bench(name, steps=150):
    torch.manual_seed(0)
    model = BionicLLMv8(tok, hid_cap=256, hid_init=64, n_layer=4, n_embd=128,
                        block_size=128, n_head=4, head_dim=32,
                        dropout=0.1, seed=0).to(DEVICE)
    eng = DevelopmentEngine(model, tok, tr, va, lr=1e-3, block_size=128,
                            n_batch=16, warmup_steps=40)
    samples.clear()
    _stop.clear()
    th = threading.Thread(target=sampler)
    th.start()
    t0 = time.time()
    eng.train_loop(steps, verbose=False)
    dt = time.time() - t0
    time.sleep(1.5)          # 让采样线程多抓几帧
    _stop.set()
    th.join()
    hcus = [float(x[1].rstrip('%')) for x in samples]
    vrams = [float(x[0].rstrip('%')) for x in samples]
    hcu_s = (f"avg {sum(hcus)/len(hcus):.1f}% max {max(hcus):.1f}%"
             if hcus else "n/a")
    vram_s = f"max {max(vrams):.0f}%" if vrams else "n/a"
    print(f"[{name}] {steps} steps {dt:.1f}s -> {dt/steps*1000:.0f} ms/step"
          f" | HCU {hcu_s} | VRAM {vram_s} (n={len(samples)})", flush=True)


# ---- 1) 基线：在线分词（现状） ----
DevelopmentEngine._seq_for = _seq_for_dev
bench('baseline online-tok')

# ---- 2) 优化：预分词缓存（结果等价，纯提速） ----
_tok_cache = {}

def _seq_for_cached(self, texts, n):
    lines = self.rng.sample(texts, min(n, len(texts)))
    max_len = self.block_size - 1
    rows = []
    for line in lines:
        ids = _tok_cache.get(line)
        if ids is None:
            ids = self.tok.encode(line)
            _tok_cache[line] = ids
        rows.append(self.tok.pad_to(ids[:max_len], max_len))
    X = torch.tensor([[self.tok.bos_id] + r for r in rows], dtype=torch.long)
    Y = torch.tensor([r + [self.tok.eos_id] for r in rows], dtype=torch.long)
    return X.to(DEVICE), Y.to(DEVICE)

DevelopmentEngine._seq_for = _seq_for_cached
bench('cached-tok')

f2, _ = torch.cuda.mem_get_info()
print(f"[mem-after] free {f2/2**30:.2f} GB", flush=True)
print('[done]')
