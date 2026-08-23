# -*- coding: utf-8 -*-
"""Bionic LLM v7 —— 异构神经元全量语言模型

人脑启示（对齐老板的"人脑神经元很多、很复杂、很多样"）：
  1. 神经元多样（heterogeneity）：皮层不同区域 = 不同细胞类型（锥体/中间神经元/浦肯野…）
     -> v7 每层 transformer 块参数各不相同：
        - 异构激活函数（ReLU / GELU / SiLU / Mish / PReLU 混合，类比不同放电模式）
        - 异构 FFN 宽度（2x/3x/4x/5x/6x n_embd，类比不同胞体大小/分支密度）
        - 异构注意力 head 数（层间不同 head 划分 = 不同感受野/功能柱）
        - 异构 RoPE 基频（不同尺度位置敏感度，类比不同时间常数神经元）
        - 异构 head 维度（head_dim 不整除时零填充投影）
        - 异构归一化（RMSNorm / LayerNorm 混合，类比不同胞体膜电导）
  2. 神经元增益可塑（gain scaling）：每层可学习的 per-dim 增益，仿"突触可塑性缩放"
  3. 大规模：支持约 1000 万参数级（1440 万），对比 v6 同构（190 万）
  4. 训练算法升级（配合更大网络）：
     - warmup + cosine 学习率（LLM 标准）
     - 权重 EMA（指数移动平均，仿突触稳态）
     - qk-norm（注意力稳定）
     - 每层异构 dropout（深层少 drop，浅层多 drop）
"""
import os
import json
import math
import time
import random
import copy
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

SEED = 0
DEVICE = 'cpu'


# ============================================================================
# 1. BPE Tokenizer（字符级，中英文覆盖）
# ============================================================================
class BPETokenizer:
    def __init__(self, vocab_cn=600, seed=SEED):
        self.rng = random.Random(seed)
        base = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 .,:;!?()[]+-*/=<>\"'"
        )
        cn = (
            "的一是了我不人在他有这上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
            "明气力或地新内大例真成果总平全家人月作得十后分小无开手又见么别给认各儿产边问话去风公功四让身才二几两由它了一什么于自少军直决式论长定传深口非常正界府感代身发问任你孩"
            "父句真正员报面家热手脑让台机票钱带走既站认准确满各略慢存信速安全网络病毒木马密码账号登录注册下载安装软件程序文件数据备份恢复清理扫描查杀更新补丁"
        )
        self.chars = list(dict.fromkeys(list(base) + list(cn)))
        self.itos = ['<pad>', '<bos>', '<eos>', '<unk>'] + self.chars[:vocab_cn]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3

    @property
    def vocab_size(self):
        return len(self.itos)

    def get_vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(ch, self.unk_id) for ch in str(text)]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids if i not in (0, 1, 2))

    def pad_to(self, ids, n):
        ids = list(ids)[:n]
        return ids + [self.pad_id] * (n - len(ids))

    def fit(self, texts, max_vocab=2500):
        """从语料构建字符词表：保证所有出现字符可编码，消除 <unk>"""
        all_chars = set()
        for t in texts:
            all_chars.update(str(t))
        freq = {}
        for t in texts:
            for ch in str(t):
                freq[ch] = freq.get(ch, 0) + 1
        ordered = sorted(all_chars, key=lambda c: (-freq.get(c, 0), c))
        base = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
        ordered = [c for c in ordered if c in base] + \
                  [c for c in ordered if c not in base]
        keep = ordered[:max_vocab]
        self.itos = ['<pad>', '<bos>', '<eos>', '<unk>'] + keep
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        return self

    def __len__(self):
        return len(self.itos)


# ============================================================================
# 2. 异构激活函数（神经元放电模式多样性）
# ============================================================================
class HeteroAct(nn.Module):
    """按层配置不同激活函数（神经元放电模式多样性）"""
    _names = ('relu', 'gelu', 'silu', 'prelu', 'mish', 'tanh')

    def __init__(self, kind, dim=None):
        super().__init__()
        self.kind = kind
        if kind == 'prelu':
            self.prelu = nn.PReLU(1)   # 标量增益，跨所有通道共享（仿单一阈值可塑）
        elif kind == 'mish':
            pass

    def forward(self, x):
        if self.kind == 'relu':
            return F.relu(x)
        if self.kind == 'gelu':
            return F.gelu(x, approximate='tanh')
        if self.kind == 'silu':
            return F.silu(x)
        if self.kind == 'prelu':
            return self.prelu(x)
        if self.kind == 'mish':
            return x * torch.tanh(F.softplus(x))
        if self.kind == 'tanh':
            return torch.tanh(x)
        return F.relu(x)


# ============================================================================
# 3. 异构归一化
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.ln(x)


# ============================================================================
# 4. 异构 RoPE 位置编码（每层不同基频 = 不同时间常数）
# ============================================================================
def precompute_rope(head_dim, max_seq, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    return freqs


def apply_rope(x, freqs):
    """x: (B, T, H, D)。freqs: (T, D//2)。"""
    B, T, H, D = x.shape
    x = x.reshape(B, T, H, D // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    f = freqs[:T].to(x.device).view(1, T, 1, -1)
    cos, sin = torch.cos(f), torch.sin(f)
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack([out0, out1], dim=-1).reshape(B, T, H, D)


def _pad_even(x):
    """若 head_dim 为奇数，末尾补 0 使其为偶数（RoPE 需两两配对）"""
    if x.size(-1) % 2 == 1:
        x = F.pad(x, (0, 1))
    return x


# ============================================================================
# 5. 异构注意力（head 数/head 维度/RoPE 基频可逐层不同）
# ============================================================================
class HeterAttention(nn.Module):
    """异构注意力：n_head / head_dim / RoPE 基频可逐层不同"""
    def __init__(self, n_embd, n_head, head_dim, block_size, base=10000.0,
                 qk_norm=False, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim          # 本层 head 维度（可 != n_embd//n_head）
        self.block_size = block_size
        self.qk_norm = qk_norm
        self.qkv = nn.Linear(n_embd, 3 * n_head * head_dim, bias=False)
        self.proj = nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
        self.register_buffer("freqs", precompute_rope(head_dim, block_size, base))
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_head * self.head_dim, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        q = _pad_even(q)
        k = _pad_even(k)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, self.freqs)
        k = apply_rope(k, self.freqs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.proj(y)


# ============================================================================
# 6. 异构 FFN（宽度/激活可逐层不同）
# ============================================================================
class HeteroMLP(nn.Module):
    def __init__(self, n_embd, mult, act_kind, dropout=0.0, gain=1.0):
        super().__init__()
        hid = int(n_embd * mult)
        self.fc1 = nn.Linear(n_embd, hid)
        self.act = HeteroAct(act_kind, dim=hid)
        self.fc2 = nn.Linear(hid, n_embd)
        self.gain = nn.Parameter(torch.full((1,), float(gain)))  # 神经元增益
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(self.drop(h)) * torch.clamp(self.gain, 0.05, 5.0)


# ============================================================================
# 7. 异构 Block（每层参数全部不同）
# ============================================================================
class HeteroBlock(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, ff_mult, act_kind, norm_kind,
                 rope_base, block_size, qk_norm=False, dropout=0.0, gain=1.0):
        super().__init__()
        self.norm1 = LayerNorm(n_embd) if norm_kind == 'ln' else RMSNorm(n_embd)
        self.attn = HeterAttention(n_embd, n_head, head_dim, block_size,
                                  base=rope_base, qk_norm=qk_norm, dropout=dropout)
        self.norm2 = LayerNorm(n_embd) if norm_kind == 'ln' else RMSNorm(n_embd)
        self.mlp = HeteroMLP(n_embd, ff_mult, act_kind, dropout=dropout, gain=gain)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 8. BionicLLM v7
# ============================================================================
class BionicLLMv7(nn.Module):
    """异构神经元全量 LLM：N 层异构 Block + 异构嵌入/输出

    支持两种模式：
      - heterogeneous: 每层独立配置（异构）
      - homogeneous:   所有层用同一配置（对齐 baseline，公平对比）
    """

    def __init__(self, tok, n_layer=8, n_embd=384, block_size=128, dropout=0.1,
                 seed=SEED, heterogeneous=True, layer_cfg=None):
        super().__init__()
        torch.manual_seed(seed)
        self.tok = tok
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.heterogeneous = heterogeneous

        if layer_cfg is None:
            layer_cfg = self._default_cfg(n_layer, heterogeneous)

        self.token_emb = nn.Embedding(tok.vocab_size, n_embd)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList()
        for cfg in layer_cfg:
            self.blocks.append(HeteroBlock(
                n_embd, cfg['n_head'], cfg['head_dim'], cfg['ff_mult'],
                cfg['act'], cfg['norm'], cfg['rope_base'], block_size,
                qk_norm=cfg.get('qk_norm', False),
                dropout=cfg.get('dropout', dropout),
                gain=cfg.get('gain', 1.0)))
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, tok.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.layer_cfg = layer_cfg
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    @staticmethod
    def _default_cfg(n_layer, heterogeneous):
        """层配置。heterogeneous=True 时每层不同（仿神经元多样性）；
        False 时全部相同（baseline）。"""
        if not heterogeneous:
            return [dict(n_head=4, head_dim=96, ff_mult=4, act='relu',
                         norm='rms', rope_base=10000.0, qk_norm=False,
                         dropout=0.1, gain=1.0)] * n_layer
        acts = ['relu', 'gelu', 'silu', 'prelu', 'mish', 'tanh', 'gelu', 'relu']
        mults = [3, 4, 6, 8, 6, 4, 4, 3]
        heads = [2, 2, 4, 4, 6, 6, 8, 8]
        head_dims = [8, 16, 8, 16, 8, 16, 8, 16]
        bases = [5000.0, 10000.0, 10000.0, 20000.0, 50000.0, 10000.0, 10000.0, 5000.0]
        norms = ['rms', 'rms', 'rms', 'ln', 'ln', 'rms', 'rms', 'rms']
        qk = [True, True, False, False, False, False, True, True]
        drops = [0.15, 0.12, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03]
        gains = [0.8, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0, 0.9]
        out = []
        for i in range(n_layer):
            out.append(dict(
                n_head=heads[i % len(heads)],
                head_dim=head_dims[i % len(head_dims)],
                ff_mult=mults[i % len(mults)],
                act=acts[i % len(acts)],
                norm=norms[i % len(norms)],
                rope_base=bases[i % len(bases)],
                qk_norm=qk[i % len(qk)],
                dropout=drops[i % len(drops)],
                gain=gains[i % len(gains)]))
        return out

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"seq {T} > block {self.block_size}"
        x = self.token_emb(idx)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=self.tok.pad_id)
        return logits, loss

    def configure_optimizer(self, lr, wd=0.05, betas=(0.9, 0.95)):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        return torch.optim.AdamW(
            [{'params': decay, 'weight_decay': wd},
             {'params': no_decay, 'weight_decay': 0.0}],
            lr=lr, betas=betas)

    def params_vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()]).float()

    def set_params_vector(self, vec):
        i = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(vec[i:i + n].reshape(p.shape))
                i += n

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# 9. 神经调制（DA/ACh/NE + TM-R）
# ============================================================================
class Neuromodulator:
    def __init__(self, ne_patience=8, ne_decay=0.8):
        self.da = 0.0
        self.ach_phase = 'learn'
        self.ne = 0.0
        self.ne_patience = ne_patience
        self.ne_decay = ne_decay
        self.stagnant = 0
        self.prev_loss = None
        self.best_loss = 1e9
        self.value_fast = None
        self.value_slow = None
        self.tmrl = 0.0

    def update(self, loss, gen):
        if self.prev_loss is None:
            self.prev_loss = loss
            self.value_fast = loss
            self.value_slow = loss
        self.value_fast = 0.7 * self.value_fast + 0.3 * loss
        self.value_slow = 0.95 * self.value_slow + 0.05 * loss
        self.tmrl = (self.value_slow - self.value_fast) / (self.value_slow + 1e-12)
        imp = self.prev_loss - loss
        self.da = imp
        if self.value_fast < self.value_slow:
            self.stagnant = max(0, self.stagnant - 2)
        else:
            self.stagnant += 1
        patience = self.ne_patience
        if self.tmrl < -0.02:
            patience = max(2, int(patience * 0.7))
        if self.stagnant >= patience:
            self.ne = min(1.0, self.ne + 0.15)
        else:
            self.ne *= self.ne_decay
        self.ach_phase = 'eval' if (gen % 3 == 2 and self.da < 0.005) else 'learn'
        if self.ne > 0.5:
            self.ach_phase = 'learn'
        self.prev_loss = loss

    def lr_mod(self, base_lr):
        lr = base_lr
        if self.ach_phase == 'eval':
            lr *= 0.5
        if 0.3 < self.ne <= 0.7:
            lr *= 0.8
        if self.da > 0.02:
            lr *= 1.15
        return lr


# ============================================================================
# 10. 自进化引擎 v7（warmup+cosine / EMA / 神经调制）
# ============================================================================
class SelfEvolutionEngineV7:
    def __init__(self, model, tok, train_texts, val_texts, seed=0,
                 lr=1e-3, block_size=128, n_batch=16, wd=0.05,
                 curriculum=True, warmup_steps=60, use_ema=True,
                 ema_decay=0.998):
        self.model = model
        self.tok = tok
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.rng = random.Random(seed)
        self.lr = lr
        self.block_size = block_size
        self.n_batch = n_batch
        self.wd = wd
        self.curriculum = curriculum
        self.warmup_steps = warmup_steps
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.mod = Neuromodulator()
        self.optimizer = None
        self._best_vec = None
        self._best_loss = 1e9
        self._ema_vec = None
        self.hist = {'steps': [], 'loss': [], 'val': [], 'ne': [], 'da': [],
                     'bbl': [], 'lr': [], 'ach': []}
        self.train_short = sorted([t for t in train_texts if len(t) <= 12], key=len)
        self.train_long = sorted([t for t in train_texts if len(t) > 12], key=len)

    # ---- 数据 ----
    def _seq_for(self, texts, n):
        lines = self.rng.sample(texts, min(n, len(texts)))
        max_len = self.block_size - 1
        rows = []
        for line in lines:
            ids = self.tok.encode(line)[:max_len]
            rows.append(self.tok.pad_to(ids, max_len))
        X = torch.tensor([[self.tok.bos_id] + r for r in rows], dtype=torch.long)
        Y = torch.tensor([r + [self.tok.eos_id] for r in rows], dtype=torch.long)
        return X, Y

    def _val_curriculum(self, gen):
        if not self.curriculum:
            return self.train_texts
        if gen < 10:
            return self.train_short
        frac = min(1.0, (gen - 10) / 30.0)
        n_long = int(len(self.train_long) * frac)
        return self.train_short + self.train_long[:n_long]

    def _val_loss(self):
        self.model.eval()
        with torch.no_grad():
            X, Y = self._seq_for(self.val_texts, len(self.val_texts))
            _, loss = self.model(X, Y)
        self.model.train()
        return float(loss.item())

    # ---- 学习率调度：warmup + cosine（标准 LLM 训练） ----
    def _lr_schedule(self, step, total):
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        p = (step - self.warmup_steps) / max(1, total - self.warmup_steps)
        return self.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    # ---- 单步 ----
    def step(self, step, total_steps):
        texts = self._val_curriculum(step)
        X, Y = self._seq_for(texts, self.n_batch)
        self.model.train()
        _, loss = self.model(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # EMA（慢权重）
        if self.use_ema:
            vec = self.model.params_vector()
            if self._ema_vec is None:
                self._ema_vec = vec.clone()
            else:
                self._ema_vec = self.ema_decay * self._ema_vec + (1 - self.ema_decay) * vec

        val = self._val_loss()
        self.mod.update(val, step)
        lr_eff = self.mod.lr_mod(self._lr_schedule(step, total_steps))
        for g in self.optimizer.param_groups:
            g['lr'] = lr_eff

        # 最优保存（按验证损失）
        if val < self._best_loss:
            self._best_loss = val
            self._best_vec = self.model.params_vector().clone()

        self.hist['steps'].append(step)
        self.hist['loss'].append(loss.item())
        self.hist['val'].append(val)
        self.hist['lr'].append(lr_eff)
        self.hist['ne'].append(self.mod.ne)
        self.hist['da'].append(self.mod.da)
        self.hist['ach'].append(1 if self.mod.ach_phase == 'learn' else 0)
        return loss.item(), val

    def load_ema(self):
        """恢复 EMA 慢权重（突触稳态）"""
        if self._ema_vec is not None:
            self.model.set_params_vector(self._ema_vec)

    # ---- 训练 ----
    def train_loop(self, n_steps, verbose=True, out_path=None):
        self.optimizer = self.model.configure_optimizer(self.lr, self.wd)
        t0 = time.time()
        for i in range(n_steps):
            loss, val = self.step(i, n_steps)
            if verbose and (i % 10 == 0 or i == n_steps - 1):
                print(f"  [步 {i:>4}/{n_steps}] loss={loss:.4f} val={val:.4f} "
                      f"NE={self.mod.ne:.2f} lr={self.hist['lr'][-1]:.6f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  [完成] 训练 {n_steps} 步，耗时 {time.time()-t0:.1f}s")
        if self._best_vec is not None:
            self.model.set_params_vector(self._best_vec)
        if out_path:
            self.save(out_path)
        return self.hist

    def save(self, path):
        state = {
            'model': {k: v.cpu().tolist() for k, v in self.model.state_dict().items()},
            'hist': {k: [float(x) for x in v] for k, v in self.hist.items()},
            'vocab': self.tok.itos,
            'layer_cfg': self.layer_cfg,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=1)

    @property
    def layer_cfg(self):
        return self.model.layer_cfg


# ============================================================================
# 11. 采样生成
# ============================================================================
def generate(model, tok, prompt, max_new_tokens=64, temperature=0.8,
             top_p=0.9, repeat_penalty=1.15, seed=None):
    if seed is not None:
        random.seed(seed)
    model.eval()
    ids = tok.encode(prompt)[:model.block_size - 1]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([[tok.bos_id] + ids[-(model.block_size - 1):]],
                             dtype=torch.long)
            logits, _ = model(x)
            logits = logits[0, -1, :] / temperature
            if ids:
                for t in set(ids[-8:]):
                    if logits[t] > 0:
                        logits[t] = logits[t] / repeat_penalty
                    else:
                        logits[t] = logits[t] * repeat_penalty
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            order = np.argsort(-probs)
            cum = np.cumsum(probs[order])
            keep = order[cum <= top_p]
            if len(keep) == 0:
                keep = order[:1]
            sel = int(np.random.choice(keep, p=probs[keep] / probs[keep].sum()))
            if sel == tok.eos_id:
                break
            ids.append(sel)
    return tok.decode(ids)


# ============================================================================
# 12. 演示
# ============================================================================
def demo():
    print("Bionic LLM v7 —— 异构神经元全量模型")
    print("=" * 60)
    tok = BPETokenizer()
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(here, '.temp', 'corpus', 'corpus_zh_en.txt')
    with open(corpus_path, encoding='utf-8') as f:
        texts = [ln.strip() for ln in f if ln.strip()]
    tok.fit(texts)
    print(f"corpus: {len(texts)} lines, vocab: {len(tok)}")
    random.Random(0).shuffle(texts)
    split = int(len(texts) * 0.85)
    tr, va = texts[:split], texts[split:]

    model = BionicLLMv7(tok, n_layer=6, n_embd=192, block_size=128,
                        heterogeneous=True)
    print(f"params: {model.n_params:,}")
    print("层配置: " + json.dumps(model.layer_cfg, ensure_ascii=False))
    eng = SelfEvolutionEngineV7(model, tok, tr, va, lr=1e-3, block_size=128,
                                n_batch=16, warmup_steps=100)
    eng.train_loop(300, out_path='.temp/llm_v7_demo.json')

    print("\n[演示] 生成示例：")
    for p in ["3+4=", "The capital of France is", "7*8="]:
        out = generate(model, tok, p, max_new_tokens=30)
        print(f"  prompt: {p!r} -> {out!r}")


if __name__ == '__main__':
    demo()