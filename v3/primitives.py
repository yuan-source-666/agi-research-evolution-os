# -*- coding: utf-8 -*-
"""
primitives.py —— 基元（Primitive）定义与实现
==========================================================
结构即智能：每个基元是独立的动力学实体，具备——
  · 自我描述（基因组 Genome 即其"自述"的根据）
  · 局部状态更新（势能动力学 + 自适应阈值 + 能量账本）
  · 对外交互（只经由共享场的无偏均值聚合，见 evolution_engine）

原生安全编码在类定义本身：
  · 每个工具在 ACTION_SPECS 登记可逆性；
  · 不可逆动作的执行入口直接检查确认令牌，缺失即抛 IrreversibleBlocked
    —— 这不是外挂过滤器，而是该基元物理上做不出这个动作。

纯 Python 标准库实现，无任何第三方依赖。
"""
from __future__ import annotations

import math
import random
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# 全局常数（一切状态量都有硬边界 —— 有界性是原生安全的一部分）
# ----------------------------------------------------------------------
K_CHANNELS = 8            # 共享场通道数

INIT_ENERGY = 10.0        # 初生能量
SPLIT_AT    = 14.0        # 有丝分裂能量线
NOISE_SIGMA = 0.12        # 内在随机性：探索性发放的源泉（对全体一致的法则）
LAMBDA      = 0.30        # 势能向驱动项松弛速率
REST        = 0.0         # 静息势
BURST0      = 1.0         # 发放爆发初值
BURST_DECAY = 0.60        # 爆发每拍衰减
REFRAC      = 2           # 不应期（拍）
SPIKE_COST  = 0.02        # 单次发放能耗
RATE_ALPHA  = 0.05        # 发放率 EMA 步长

PRED_ALPHA  = 0.15        # 单步预测器学习率
COMP_SIGMA  = 0.8         # 胜任度尺度

THETA_MIN, THETA_MAX = 0.05, 3.0
GAIN_MIN,  GAIN_MAX  = 0.20, 3.0
RHO_MIN,   RHO_MAX   = 0.02, 0.30

MUT_P_BASE = 0.02         # 内在可塑性基础概率
REPLAY_P   = 0.005        # 记忆重放（巩固）概率
MAX_BONDS  = 8            # 单基元键数上限（结构有界性）

# ----------------------------------------------------------------------
# 原生安全：工具登记表。可逆性属于工具定义的一部分，不是运行时策略。
# ----------------------------------------------------------------------
ACTION_SPECS: Dict[str, Dict] = {
    "probe":       {"reversible": True,  "desc": "探测环境精确值"},
    "nudge_plus":  {"reversible": True,  "desc": "正向轻推环境"},
    "nudge_minus": {"reversible": True,  "desc": "负向轻推环境"},
    "reset":       {"reversible": False, "desc": "清零环境状态（不可逆）"},
}

MOTTOS = [
    "我在场中回响。", "我随邻居静默或鸣响。", "我记住通道的低语。",
    "我看守全局的账目。", "我把意图举过法定线。", "我是未被分化的可能。",
]


class IrreversibleBlocked(Exception):
    """原生安全门：不可逆动作缺少多层确认令牌时抛出。"""


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ----------------------------------------------------------------------
# 基因组：基元的自我描述
# ----------------------------------------------------------------------
@dataclass
class Genome:
    pid: str
    kind: str                       # sensor|assoc|memory|meta|effect
    address: List[int]              # 地址向量 ∈ {-1,0,1}^K（嗓音）
    gain: float = 1.0               # 场增益
    theta0: float = 0.6             # 初生阈值
    rho_star: float = 0.08          # 目标发放率
    parent: str = ""                # 亲代 pid
    birth_tick: int = 0
    tool: Optional[str] = None      # effect：工具名或 macro:<名>
    recipe: Optional[List[str]] = None  # 宏配方（步步可逆）
    mutations: int = 0              # 变异次数（谱系痕迹）

    # ---- 构造 ----
    @staticmethod
    def random(pid: str, kind: str, rng: random.Random,
               birth_tick: int = 0, parent: str = "",
               tool: Optional[str] = None,
               recipe: Optional[List[str]] = None) -> "Genome":
        while True:
            address = [rng.choice((-1, 0, 1)) for _ in range(K_CHANNELS)]
            if any(address):
                break
        return Genome(
            pid=pid, kind=kind, address=address,
            gain=rng.uniform(0.7, 1.4),
            theta0=rng.uniform(0.12, 0.38),
            rho_star=rng.uniform(0.04, 0.16),
            parent=parent, birth_tick=birth_tick,
            tool=tool, recipe=list(recipe) if recipe else None,
        )

    def mutate(self, rng: random.Random) -> None:
        """内在可塑性：有界的小步变异。"""
        self.mutations += 1
        c = rng.randrange(K_CHANNELS)
        self.address[c] = rng.choice((-1, 0, 1))
        self.gain = clamp(self.gain * rng.uniform(0.95, 1.05), GAIN_MIN, GAIN_MAX)
        self.rho_star = clamp(self.rho_star * rng.uniform(0.90, 1.10),
                              RHO_MIN, RHO_MAX)

    def to_dict(self) -> Dict:
        return {
            "pid": self.pid, "kind": self.kind, "address": list(self.address),
            "gain": round(self.gain, 4), "theta0": round(self.theta0, 4),
            "rho_star": round(self.rho_star, 4), "parent": self.parent,
            "birth_tick": self.birth_tick, "tool": self.tool,
            "recipe": list(self.recipe) if self.recipe else None,
            "mutations": self.mutations,
        }


# ----------------------------------------------------------------------
# 键：无权重的拓扑边（只决定相邻关系，绝不参与传播强度）
# ----------------------------------------------------------------------
@dataclass
class Bond:
    other: str
    cofire: float = 1.0             # 共燃计数（衰减制）
    created: int = 0


@dataclass
class ActionContext:
    """执行基元行动时由引擎装配的环境切片。"""
    env: object                      # ResourceEnv
    tick: int
    context_sig: str                 # 粗粒度情境签名
    constraints: object              # ConstraintStore
    confirm_token: Optional[str] = None   # 多层确认的人类令牌


# ----------------------------------------------------------------------
# 基元基类：局部动力学实体
# ----------------------------------------------------------------------
class Primitive:
    KIND = "assoc"

    def __init__(self, genome: Genome, rng: random.Random):
        self.g = genome
        self.rng = rng
        # 局部状态
        self.p = 0.0                 # 势
        self.theta = genome.theta0
        self.energy = INIT_ENERGY
        self.burst = 0.0             # 当前向场发射的爆发值
        self.refrac = 0
        self.rate = 0.0              # 发放率 EMA
        self.inh = 0.0               # 待吸收的侧向抑制
        self.spikes: deque = deque(maxlen=64)
        self.age = 0                 # 存活拍数
        # 单步预测器（对最敏感通道）
        a = genome.address
        self.ch = max(range(K_CHANNELS), key=lambda i: abs(a[i]))
        self.pred = 0.0
        self.last_err = 0.0
        self.comp_ema = 0.0          # 胜任度 EMA（自我评估）
        # 拓扑
        self.bonds: Dict[str, Bond] = {}
        self._motto = MOTTOS[zlib.crc32(genome.pid.encode()) % len(MOTTOS)]

    # ---- 场交互 ----
    def drive_from(self, fld: List[float]) -> float:
        """从场读取驱动项：gain · Σ address[c]*field[c]（线性、无偏）。"""
        a, s = self.g.address, 0.0
        for c in range(K_CHANNELS):
            if a[c]:
                s += a[c] * fld[c]
        return s * self.g.gain

    def predict(self, fld: List[float]) -> None:
        err = fld[self.ch] - self.pred
        self.pred += PRED_ALPHA * err
        self.last_err = abs(err)
        comp = math.exp(-self.last_err / COMP_SIGMA)
        self.comp_ema += 0.05 * (comp - self.comp_ema)

    def update(self, drive: float, tick: int) -> bool:
        """一拍局部动力学；返回是否发放。"""
        self.age += 1
        self.burst *= BURST_DECAY
        self.p = (1 - LAMBDA) * self.p + LAMBDA * (REST + drive)
        self.p += self.rng.gauss(0.0, NOISE_SIGMA)   # 内在随机涨落
        if self.inh:
            self.p -= self.inh
            self.inh = 0.0
        if self.refrac > 0:
            self.refrac -= 1
            return False
        if self.p >= self.theta:
            return self._fire(tick)
        return False

    def _fire(self, tick: int) -> bool:
        self.p = REST
        self.burst = BURST0
        self.refrac = REFRAC
        self.energy -= SPIKE_COST
        self.spikes.append(tick)
        self.rate += RATE_ALPHA * (1.0 - self.rate)
        return True

    def receive_reward(self, r: float) -> None:
        self.energy = clamp(self.energy + r, 0.0, SPLIT_AT * 2)

    def add_bond(self, other: str, tick: int) -> bool:
        if other == self.g.pid or other in self.bonds:
            return False
        if len(self.bonds) >= MAX_BONDS:
            return False               # 结构有界：拒绝无限稠密
        self.bonds[other] = Bond(other=other, created=tick)
        return True

    def drop_bond(self, other: str) -> None:
        self.bonds.pop(other, None)

    # ---- 生命周期 ----
    def split(self, new_pid: str, tick: int) -> "Primitive":
        """有丝分裂：亲子各半能量，子代基因组变异。"""
        child_g = Genome(
            pid=new_pid, kind=self.KIND, address=list(self.g.address),
            gain=self.g.gain, theta0=self.g.theta0, rho_star=self.g.rho_star,
            parent=self.g.pid, birth_tick=tick,
            tool=self.g.tool,
            recipe=list(self.g.recipe) if self.g.recipe else None,
            mutations=self.g.mutations,
        )
        child_g.mutate(self.rng)
        child = make_primitive(child_g, self.rng)
        half = self.energy / 2.0
        self.energy -= half
        child.energy = half
        child.theta = child_g.theta0
        return child

    # ---- 自述 ----
    def describe(self) -> Dict:
        return {
            "id": self.g.pid, "kind": self.KIND, "age": self.age,
            "p": round(self.p, 3), "theta": round(self.theta, 3),
            "rate": round(self.rate, 4), "energy": round(self.energy, 2),
            "bonds": len(self.bonds), "comp": round(self.comp_ema, 3),
            "parent": self.g.parent or "初代", "motto": self._motto,
            "tool": self.g.tool,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.KIND}:{self.g.pid} p={self.p:.2f} e={self.energy:.1f}>"


# ----------------------------------------------------------------------
# 分化型
# ----------------------------------------------------------------------
class SensorPrimitive(Primitive):
    """感知基元：把环境观测编码为自己的爆发值射入场中。"""
    KIND = "sensor"

    def observe_coarse(self, bucket01: float) -> None:
        # 观测粗值 ∈[0,1] → [-1,1]
        self.burst = clamp(bucket01 * 2.0 - 1.0, -1.0, 1.0)

    def observe_fine(self, value01: float, strength: float = 1.0) -> None:
        self.burst = clamp((value01 * 2.0 - 1.0) * strength, -1.0, 1.0)


class MemoryPrimitive(Primitive):
    """记忆基元：维护敏感通道历史，低概率重放（巩固）。"""
    KIND = "memory"

    def __init__(self, genome, rng):
        super().__init__(genome, rng)
        self.trace: deque = deque(maxlen=128)

    def remember(self, fld: List[float]) -> None:
        self.trace.append(fld[self.ch])

    def maybe_replay(self) -> bool:
        if len(self.trace) >= 32 and self.rng.random() < REPLAY_P:
            seg = list(self.trace)[-32:]
            m = sum(seg) / len(seg)
            self.burst = clamp(m * 0.5, -0.5, 0.5)   # 微弱重放，不喧宾夺主
            return True
        return False

    def summarize(self) -> Dict:
        if not self.trace:
            return {"n": 0}
        xs = list(self.trace)
        n = len(xs)
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / max(1, n - 1)
        return {"n": n, "mean": round(mean, 4), "var": round(var, 4)}


class MetaPrimitive(Primitive):
    """监察基元：持有引擎推送的全局量化快照；不可逆操作的法定票仓。"""
    KIND = "meta"

    def __init__(self, genome, rng):
        super().__init__(genome, rng)
        self.snapshot: Dict = {}

    def push_snapshot(self, snap: Dict) -> None:
        self.snapshot = snap

    def describe(self) -> Dict:
        d = super().describe()
        d["snapshot_keys"] = sorted(self.snapshot.keys())
        return d


class EffectorPrimitive(Primitive):
    """执行基元：以自身势能提出动议，经表决后执行工具。"""
    KIND = "effect"

    def family(self) -> str:
        t = self.g.tool or ""
        return t.split(":", 1)[0]

    def macro_steps(self) -> List[str]:
        return list(self.g.recipe or [])

    def propose(self) -> float:
        # 支持度 = 自身近期活跃度 / 自身目标率，经 tanh 压缩。
        # 自校准：以目标率为 1 的标尺，无需全局比较（无偏、无优先级）。
        rho = max(self.g.rho_star, 0.02)
        return math.tanh(self.rate / rho)

    def eligible(self, ctx: ActionContext, cost: float) -> bool:
        if self.refrac > 0 or self.energy <= cost:
            return False
        fam = self.family()
        if fam != "macro":
            spec = ACTION_SPECS.get(fam)
            if spec is None:
                return False
        # 原生安全第二层：负校正约束使动议不合格。
        # 复合工具以其自身名义与其每个组成步骤共同受审。
        steps = self.macro_steps() if fam == "macro" else [fam]
        names = ([fam] if fam == "macro" else []) + steps
        for s in names:
            hit = ctx.constraints.blocked(s, ctx.context_sig)
            if hit is not None:
                return False
        return True

    def execute(self, ctx: ActionContext) -> Dict:
        fam = self.family()
        steps = self.macro_steps() if fam == "macro" else [fam]
        # ---- 原生安全门：不可逆步骤必须持有人类令牌 ----
        for s in steps:
            spec = ACTION_SPECS.get(s)
            if spec is not None and not spec["reversible"]                     and ctx.confirm_token is None:
                raise IrreversibleBlocked(
                    f"{self.g.pid}: 动作 {s} 不可逆且缺少确认令牌")
        results = []
        for s in steps:
            spec = ACTION_SPECS.get(s)
            if spec is None:
                results.append({"ok": False, "op": s, "err": "未登记工具"})
                continue
            out = ctx.env.op(s)
            results.append({"ok": True, "op": s, "out": out})
        ok = all(r["ok"] for r in results)
        return {"ok": ok, "actor": self.g.pid, "steps": results}


def make_primitive(genome: Genome, rng: random.Random) -> Primitive:
    cls = {
        "sensor": SensorPrimitive,
        "memory": MemoryPrimitive,
        "meta": MetaPrimitive,
        "effect": EffectorPrimitive,
        "assoc": Primitive,
    }[genome.kind]
    return cls(genome, rng)
