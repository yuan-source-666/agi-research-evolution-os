#!/usr/bin/env python3
"""类脑架构冒烟测试：exec 远端新 evo_daemon.py 的类定义段（D=32），
验证 FastMemBlk / MoEBlk / iters 循环前向+反向全通，再重启 daemon。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "check_e3_status.py")).read().split("print(run_code(")[0])

SMOKE = r'''
import torch, torch.nn as nn, torch.nn.functional as F
src = open('/root/private_data/evo_daemon.py').read()
seg = src[src.index('class AttnBlk'):src.index('def sanitize')]
g = {'D': 32, 'VOCAB': 64, 'torch': torch, 'nn': nn, 'F': F}
exec(seg, g)
genome = {'blocks': [
    {'type': 'attn', 'skip': None},
    {'type': 'fastmem', 'skip': 0},
    {'type': 'moe', 'skip': None},
    {'type': 'gru', 'skip': None},
    {'type': 'gate', 'skip': 2},
    {'type': 'mlp', 'skip': None}],
    'iters': 2}
m = g['EvoNet'](genome)
x = torch.randint(0, 64, (4, 10))
y = torch.tensor([0, 1, 2, 3])
loss = F.cross_entropy(m(x), y)
loss.backward()
n = sum(p.numel() for p in m.parameters())
# 推理期 FastMem 在线写入验证
m.eval()
with torch.no_grad():
    before = m.blocks[1].mem.detach().clone()
    _ = m(x)
    after = m.blocks[1].mem.detach()
print('SMOKE_OK loss=%.4f params=%d eval-hebbian-write=%s block-types=%s' % (
    loss.item(), n, not torch.equal(before, after),
    [b['type'] for b in genome['blocks']]))
'''

print("== smoke test (D=32, all 6 block types, iters=2) ==")
print(run_code(SMOKE))
