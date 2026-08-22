# -*- coding: utf-8 -*-
"""Bionic LLM v8 —— 发育式神经网络（Developmental Transformer）

人脑启示（对齐老板的核心质疑："人脑都是自己成长的，不需要构建如此复杂"）：
  v7 的问题：异构是"设计"出来的——激活种类/head 数/FFN 宽度全部由人预设，
  复杂度来自设计蓝图，而不是来自学习本身。这与真实大脑相反：
  大脑的复杂结构是"长"出来的——由简单的发育规则 + 使用经验逐步塑造。

v8 的核心主张：
  >>> 同构简单骨架 + 生长算子，让复杂度在训练中自行涌现 <<<

发育三阶段（对标大脑发育）：
  P0 过度生长（proliferation）：网络刚起步，允许高频"神经元分裂"
      —— 高负载神经元（health 强）复制出带微小扰动的子神经元，父子各得一半学习率
      （学习率缩放 = 分化：两个副本开始朝不同分工演化）
  P1 竞争修剪（pruning）：验证损失改善停滞 → 停止分裂，开始"用进废退"
      —— 长期不活跃（health 极弱）的神经元被掩码掉，权重清零（神经达尔文主义）
  P2 定型（consolidation）：结构冻结，只做权重精调（配合 EMA 慢权重 = 突触稳态）

实现机制：
  - GrowthMLP：预分配容量矩阵 + 布尔掩码（神经元是否存活）
      mask=1 参与前向/反向，mask=0 完全冻结（梯度为零）
      分裂：free slot 0→1，复制父权重+噪声；修剪：1→0，权重清零
  - 全局"信号分子"：复用 v7 的 Neuromodulator（DA/NE/ACh）驱动阶段切换时机
  - 结构可观测：每步记录有效神经元数 / 分裂数 / 修剪数 / 阶段

与 v7 的关系：复用 tokenizer / 归一化 / RoPE / 注意力 / 生成 / 神经调制基础件；
  异构不再被预设，而是发育过程的自然产物（长出来的异构才配叫异构）。
"""
import os
import json
import math
import time
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from bionic_llm_v7 import (BPETokenizer, RMSNorm, LayerNorm, precompute_rope,
                           apply_rope, _pad_even, Neuromodulator, generate,
                           HeterAttention, HeteroBlock, BionicLLMv7)

SEED = 0
DEVICE = 'cpu'


# ============================================================================
# 1. 可生长 MLP（神经元分裂 / 修剪的载体）
# ============================================================================
class GrowthMLP(nn.Module):
    """发育式 FFN：

    预分配容量 hid_cap，初始只激活 init_hid 个神经元（掩码控制）。
    - fc1: (cap, d)  每行 = 一个隐藏"神经元"的输入突触
    - fc2: (d, cap)  每列 = 该神经元的输出突触
    - mask:  神经元是否存活（True=激活参与计算）
    - health: 使用强度 EMA（激活绝对值均值），分裂/修剪的依据
    - born_at: 出生步数（判断成熟度，防止婴儿神经元被立刻修剪）
    - lr_scale: 每神经元学习率缩放（分裂后父子各减半 = 分化）
    """

    def __init__(self, d, hid_cap, init_hid, activation='gelu', dropout=0.0,
                 lr_scale_init=1.0):
        super().__init__()
        assert init_hid <= hid_cap
        self.d = d
        self.hid_cap = hid_cap
        self.activation = activation
        self.fc1 = nn.Parameter(torch.zeros(hid_cap, d))
        self.fc2 = nn.Parameter(torch.zeros(d, hid_cap))
        self.register_buffer('mask', torch.zeros(hid_cap, dtype=torch.bool))
        self.register_buffer('health', torch.zeros(hid_cap))
        self.register_buffer('act_ema', torch.zeros(hid_cap))
        self.register_buffer('born_at', torch.full((hid_cap,), -1, dtype=torch.long))
        self.register_buffer('lr_scale', torch.full((hid_cap,), float(lr_scale_init)))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 初始化：均匀小权重，只激活前 init 个
        nn.init.normal_(self.fc1, 0.0, 0.02)
        nn.init.normal_(self.fc2, 0.0, 0.02)
        with torch.no_grad():
            self.mask[:init_hid] = True
            self.born_at[:init_hid] = 0

    # ---- 前向：mask 决定哪些神经元参与 ----
    def forward(self, x):
        B, T, _ = x.shape
        w1 = self.fc1 * self.mask.unsqueeze(1).float()      # (cap, d) 掩码行置0
        h = x @ w1.T                                        # (B, T, cap)
        if self.activation == 'relu':
            h = F.relu(h)
        elif self.activation == 'gelu':
            h = F.gelu(h, approximate='tanh')
        elif self.activation == 'silu':
            h = F.silu(h)
        else:
            h = F.relu(h)
        h = self.drop(h)
        # 激活强度 EMA（评估用）
        with torch.no_grad():
            act = h.detach().abs().mean(dim=(0, 1))
            self.act_ema = 0.9 * self.act_ema + 0.1 * act
            self.health = torch.maximum(self.act_ema, self.health * 0.995)
        out = h @ self.fc2.T                                # (B, T, d)
        return out

    # ---------------- 统计 ----
    def n_active(self):
        return int(self.mask.sum().item())

    # ---------------- 发育算子：分裂 ----------------
    def split_neurons(self, step, k=None, health_quantile=0.7, mature_after=20,
                      noise=0.02):
        """复制高负载神经元：新神经元占用空槽，继承父权重+噪声；
        父神经元 lr_scale 减半（分化：两个方向分别演化）。
        返回实际分裂个数。"""
        if k is None:
            k = max(1, int(self.n_active() * 0.08))
        alive = self.mask
        if alive.sum().item() == 0:
            return 0
        mature = (step - self.born_at) >= mature_after
        eligible = alive & mature
        if eligible.sum().item() == 0:
            return 0
        th = torch.quantile(self.health[eligible], health_quantile)
        cand_idx = torch.nonzero(eligible & (self.health >= th)).squeeze(1)
        if cand_idx.numel() == 0:
            return 0
        # 候选按 health 排序，取前 k 个
        cand_idx = cand_idx[torch.argsort(self.health[cand_idx], descending=True)[:k]]
        free = torch.nonzero(~self.mask).squeeze(1)
        if free.numel() == 0:
            return 0
        n = min(cand_idx.numel(), free.numel())
        n = int(n)
        with torch.no_grad():
            for i in range(n):
                parent = int(cand_idx[i].item())
                child = int(free[i].item())
                self.fc1[child] = self.fc1[parent] + torch.randn_like(self.fc1[parent]) * noise
                self.fc2[:, child] = self.fc2[:, parent] + torch.randn_like(self.fc2[:, parent]) * noise
                self.mask[child] = True
                self.born_at[child] = step
                self.lr_scale[child] = 0.5 * float(self.lr_scale[parent].detach())
                self.lr_scale[parent] = 0.5 * float(self.lr_scale[parent].detach())
                self.health[child] = float(self.health[parent].detach()) * 0.5
        return n

    # ---------------- 发育算子：修剪 ----------------
    def prune_neurons(self, k=None, inactive_quantile=0.2, mature_min=30):
        """淘汰长期不活跃的神经元：按 act_ema 相对分位剪掉最弱的 k 个，
        并清零其权重（神经达尔文主义：用进废退）"""
        if k is None:
            k = max(1, int(self.n_active() * 0.05))
        alive = self.mask
        if alive.sum().item() == 0:
            return 0
        eligible = alive & (self.born_at >= 0) & (self.act_ema > 0)
        if eligible.sum().item() == 0:
            return 0
        # 按 act_ema 排序，剪掉最弱的 k 个（相对分位，不依赖绝对阈值）
        cands = torch.nonzero(eligible).squeeze(1)
        cands = cands[torch.argsort(self.act_ema[cands])[:k]]
        with torch.no_grad():
            for i in range(int(cands.numel())):
                idx = int(cands[i].item())
                self.mask[idx] = False
                self.fc1[idx].zero_()
                self.fc2[:, idx].zero_()
                self.lr_scale[idx] = 0.0
                self.health[idx] = 0.0
                self.act_ema[idx] = 0.0
                self.born_at[idx] = -1
        return int(cands.numel())

    # 训练时在 backward 后按 lr_scale 缩放梯度（模拟分化后的差异化可塑性）
    def apply_lr_scale_grad(self):
        if self.fc1.grad is not None:
            self.fc1.grad.mul_(self.lr_scale.unsqueeze(1))
        if self.fc2.grad is not None:
            self.fc2.grad.mul_(self.lr_scale.unsqueeze(0))


# ============================================================================
# 2. 发育式 Block（注意力固定 + MLP 可生长）
# ============================================================================
class GrowthBlock(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, hid_cap, init_hid, block_size,
                 act='relu', dropout=0.0, rope_base=10000.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = HeterAttention(n_embd, n_head, head_dim, block_size,
                                   base=rope_base, qk_norm=False, dropout=dropout)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = GrowthMLP(n_embd, hid_cap, init_hid, activation=act,
                             dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 3. 发育控制器（三阶段：过度生长 -> 修剪 -> 定型）
# ============================================================================
class DevelopmentController:
    """阶段机：由验证损失是否刷新纪录驱动（best 久未更新 = 停滞）

    phase 0 增殖：每 split_interval 步触发一次分裂；
        当“best 超过 min_stall 步未更新”或“分裂轮数达上限”-> 转修剪
    phase 1 修剪：每 prune_interval 步触发一次修剪；
        修剪轮数达上限 -> 转定型
    phase 2 定型：结构冻结，只做权重精调
    """
    PHASE_NAMES = ['proliferation', 'pruning', 'consolidation']

    def __init__(self, split_interval=25, prune_interval=25, min_stall=60,
                 max_split_rounds=10, max_prune_rounds=3):
        self.split_interval = split_interval
        self.prune_interval = prune_interval
        self.min_stall = min_stall
        self.max_split_rounds = max_split_rounds
        self.max_prune_rounds = max_prune_rounds
        self.phase = 0
        self.best_val = 1e9
        self.best_step = 0
        self.last_split_step = 0
        self.last_prune_step = 0
        self.n_split_total = 0
        self.n_prune_total = 0
        self.phase_start_step = 0
        self.split_rounds = 0
        self.prune_rounds = 0

    def update(self, val, step):
        """每步调用：维护 best，判定阶段切换"""
        if val < self.best_val:
            self.best_val = val
            self.best_step = step
        if self.phase == 0:
            # 增殖期：长时间无新纪录 或 分裂轮数达上限 -> 转修剪
            if (step - self.best_step) >= self.min_stall or \
               self.split_rounds >= self.max_split_rounds:
                self._enter_phase(1, step)
        elif self.phase == 1:
            # 修剪期：轮数耗尽 -> 定型
            if self.prune_rounds >= self.max_prune_rounds:
                self._enter_phase(2, step)
        return self.phase

    def _enter_phase(self, p, step):
        self.phase = p
        self.phase_start_step = step

    # ---- 发育触发（由 engine 每 interval 调用）----
    def maybe_grow(self, blocks, step):
        """phase 0 时执行分裂"""
        if self.phase != 0:
            return 0
        if step - self.last_split_step < self.split_interval:
            return 0
        n = 0
        for blk in blocks:
            n += blk.mlp.split_neurons(step)
        self.last_split_step = step
        self.n_split_total += n
        self.split_rounds += 1
        return n

    def maybe_prune(self, blocks, step):
        """phase 1 时执行修剪"""
        if self.phase != 1:
            return 0
        if step - self.last_prune_step < self.prune_interval:
            return 0
        n = 0
        for blk in blocks:
            n += blk.mlp.prune_neurons()
        self.last_prune_step = step
        self.n_prune_total += n
        self.prune_rounds += 1   # 无论是否剪到都计轮数，防止卡死
        return n


# ============================================================================
# 4. 发育式语言模型 v8
# ============================================================================
class BionicLLMv8(nn.Module):
    """发育式 Transformer：同构起步，训练中长结构"""

    def __init__(self, tok, n_layer=4, n_embd=128, block_size=128,
                 n_head=4, head_dim=32, hid_cap=256, hid_init=64,
                 dropout=0.1, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.tok = tok
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.token_emb = nn.Embedding(tok.vocab_size, n_embd)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            self.blocks.append(GrowthBlock(
                n_embd, n_head, head_dim, hid_cap, hid_init, block_size,
                act='relu', dropout=dropout))
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, tok.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

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

    # ---- 结构观测 ----
    def n_active_neurons(self):
        return [blk.mlp.n_active() for blk in self.blocks]


# ============================================================================
# 5. 训练引擎 v8（发育集成）
# ============================================================================
class DevelopmentEngine:
    def __init__(self, model, tok, train_texts, val_texts, seed=0,
                 lr=1e-3, block_size=128, n_batch=16, wd=0.05,
                 warmup_steps=40, growth_interval=25,
                 curriculum=True):
        self.model = model
        self.tok = tok
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.rng = random.Random(seed)
        self.lr = lr
        self.block_size = block_size
        self.n_batch = n_batch
        self.wd = wd
        self.warmup_steps = warmup_steps

        self.curriculum = curriculum
        self.growth_interval = growth_interval
        self.val_interval = 10     # 验证频率：每 10 步评估一次（性能优化）
        self.enable_dev = True     # 发育开关：False = 固定容量基线（无分裂/修剪）
        self.mod = Neuromodulator()
        self.dev = DevelopmentController()
        self.optimizer = None
        self._best_vec = None
        self._best_loss = 1e9
        self._ema_vec = None
        self.ema_decay = 0.998
        self.use_ema = True
        self.hist = {'steps': [], 'loss': [], 'val': [], 'ne': [], 'da': [],
                     'phase': [], 'active': [], 'split': [], 'prune': [], 'lr': []}
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

    def _curriculum_texts(self, gen):
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

    def _lr_schedule(self, step, total):
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps
        p = (step - self.warmup_steps) / max(1, total - self.warmup_steps)
        return self.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    # ---- 单步 ----
    def step(self, step, total_steps):
        texts = self._curriculum_texts(step)
        X, Y = self._seq_for(texts, self.n_batch)
        self.model.train()
        _, loss = self.model(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # 神经元差异化学习率（分裂分化的核心）
        for blk in self.model.blocks:
            blk.mlp.apply_lr_scale_grad()
        self.optimizer.step()

        # EMA 慢权重
        if self.use_ema:
            vec = self.model.params_vector()
            if self._ema_vec is None:
                self._ema_vec = vec.clone()
            else:
                self._ema_vec = self.ema_decay * self._ema_vec + (1 - self.ema_decay) * vec

        # 验证损失：每 val_interval 步计算一次（val 开销大，避免每步拖慢训练）
        val = self._last_val if hasattr(self, '_last_val') else float('nan')
        if step % self.val_interval == 0 or step == total_steps - 1:
            val = self._val_loss()
            self._last_val = val
        self.mod.update(val, step)
        if self.enable_dev:
            self.dev.update(val, step)          # 阶段切换
        lr_eff = self.mod.lr_mod(self._lr_schedule(step, total_steps))
        for g in self.optimizer.param_groups:
            g['lr'] = lr_eff

        # 发育操作触发（每 interval 步）
        n_split = 0
        n_prune = 0
        if self.enable_dev and step % self.growth_interval == 0 and step > 0:
            n_split = self.dev.maybe_grow(self.model.blocks, step)
            n_prune = self.dev.maybe_prune(self.model.blocks, step)

        if val < self._best_loss:
            self._best_loss = val
            self._best_vec = self.model.params_vector().clone()

        self.hist['steps'].append(step)
        self.hist['loss'].append(loss.item())
        self.hist['val'].append(val)
        self.hist['lr'].append(lr_eff)
        self.hist['ne'].append(self.mod.ne)
        self.hist['da'].append(self.mod.da)
        self.hist['phase'].append(self.dev.phase)
        self.hist['active'].append(self.model.n_active_neurons())
        self.hist['split'].append(n_split)
        self.hist['prune'].append(n_prune)
        return loss.item(), val

    def load_ema(self):
        if self._ema_vec is not None:
            self.model.set_params_vector(self._ema_vec)

    # ---- 训练 ----
    def train_loop(self, n_steps, verbose=True, out_path=None):
        self.optimizer = self.model.configure_optimizer(self.lr, self.wd)
        t0 = time.time()
        for i in range(n_steps):
            loss, val = self.step(i, n_steps)
            if verbose and (i % 10 == 0 or i == n_steps - 1):
                act = self.model.n_active_neurons()
                print(f"  [步 {i:>4}/{n_steps}] loss={loss:.4f} val={val:.4f} "
                      f"阶段={self.dev.PHASE_NAMES[self.dev.phase]} "
                      f"神经={act} 分裂={self.dev.n_split_total} "
                      f"修剪={self.dev.n_prune_total} "
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
            'hist': {k: [x if isinstance(x, (int, float)) else list(x)
                         for x in v] for k, v in self.hist.items()},
            'vocab': self.tok.itos,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=1)


# ============================================================================
# 6. 演示
# ============================================================================
def demo():
    print("Bionic LLM v8 —— 发育式神经网络")
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

    model = BionicLLMv8(tok, n_layer=4, n_embd=128, block_size=128,
                        n_head=4, head_dim=32, hid_cap=256, hid_init=64)
    print(f"params: {model.n_params:,}  (初始神经元 {model.n_active_neurons()})")
    eng = DevelopmentEngine(model, tok, tr, va, lr=1e-3, block_size=128,
                            n_batch=16, warmup_steps=40)
    eng.train_loop(200, out_path='.temp/llm_v8_demo.json')

    print("\n[演示] 生成示例：")
    for p in ["3+4=", "The capital of France is", "7*8="]:
        out = generate(model, tok, p, max_new_tokens=30)
        print(f"  prompt: {p!r} -> {out!r}")


if __name__ == '__main__':
    demo()