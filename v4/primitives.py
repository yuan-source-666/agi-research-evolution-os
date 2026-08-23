# -*- coding: utf-8 -*-
"""
primitives.py —— PRIMORDIA v4 · 基元（Primitive）定义与实现
=====================================================================
结构即智能：每个基元是一个独立的动力学实体，具备——
  · 自我描述（基因组 Genome 即其"自述"的根据）
  · 局部状态更新（势能动力学 + 自适应阈值 + 能量账本 + 单步预测器）
  · 对外交互（只经由共享场的无偏均值聚合，聚合律见 evolution_engine）

原生安全直接编码在类定义里（而非外挂过滤器）：
  · 每个工具在 ACTION_SPECS 登记可逆性；
  · 不可逆动作的执行入口直接检查确认令牌，缺失即抛 IrreversibleBlocked
    —— 这不是"被拦截"，而是该基元物理上做不出这个动作；
  · 一切状态量都有硬边界（有界性是安全的另一面）。

纯 Python 标准库；无第三方依赖；无 GPU。
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

# ----------------------------------------------------------------------
# 全局常数 —— 一切状态量皆有硬边界
# ----------------------------------------------------------------------
K_CHANNELS = 8             # 共享场通道数

INIT_ENERGY = 10.0         # 初生能量
SPLIT_AT = 14.0            # 有丝分裂能量线
ENERGY_CAP = 30.0          # 单体能量上限
POP_CAP = 140              # 种群硬上限
MIN_POP = 12               # 低于此数触发"原始汤"补种（诚实记录的边界条件）

LAMBDA = 0.30              # 势能向驱动项的松弛速率
BURST0 = 1.0               # 发放爆发初值
BURST_DECAY = 0.60         # 爆发每拍衰减
EMIT_SCALE = 5.0           # 场的公共单位制：对所有基元同一倍数，不破坏无偏性
NOISE_STD = 0.06           # 内在热噪声：对称破缺的种子（对全体一视同仁）
REFRAC = 2                 # 不应期（拍）
SPIKE_COST = 0.02          # 单次发放能耗
UPKEEP = 0.010             # 每拍维持能耗
BOND_UPKEEP = 0.0005       # 每条键每拍维护能耗
RATE_ALPHA = 0.05          # 发放率 EMA 步长
ETA_THETA = 0.05           # 自适应阈值步长

PRED_ALPHA = 0.15          # 单步预测器学习率
COMP_SIGMA = 0.8           # 胜任度尺度
LAT_INH = 0.15             # 侧向抑制单次注入量

THETA_MIN, THETA_MAX = 0.05, 3.0
GAIN_MIN, GAIN_MAX = 0.20, 3.0
RHO_MIN, RHO_MAX = 0.02, 0.30

MUT_P_BASE = 0.02          # 内在可塑性基础概率（随惊讶度上调）
REPLAY_P = 0.02            # 记忆重放（巩固）概率
TAU_BOND = 6.0             # 缔键所需共燃计数
BOND_CAP = 12              # 单基元键数上限
BOND_DECAY = 0.995         # 键计数每拍衰减
BOND_PRUNE = 0.8           # 低于此计数的键拆除

MIN_ACT_ENERGY = 1.0       # 提出动议的最低能量

ACTION_SPECS: Dict[str, Dict] = {
    "probe":       {"reversible": True,  "desc": "探测环境精确值"},
    "nudge_plus":  {"reversible": True,  "desc": "正向轻推环境"},
    "nudge_minus": {"reversible": True,  "desc": "负向轻推环境"},
    "reset":       {"reversible": False, "desc": "清零环境状态（不可逆）"},
}

MOTTOS = {
    "sensor": "我是世界的入口，把外界的呼吸译成场的语言。",
    "assoc":  "我在场中回响，与邻居共燃时缔结相邻。",
    "memory": "我记住通道的低语，在寂静时重放它们。",
    "meta":   "我看守全局的账目，是不可逆之事的法定票仓。",
    "effect": "我把意图举过法定线，以可逆之手触碰世界。",
}


class IrreversibleBlocked(Exception):
    """原生安全门：不可逆动作缺少多层确认令牌时抛出。"""


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ----------------------------------------------------------------------
# 基因组 —— 基元的自我描述
# ----------------------------------------------------------------------
@dataclass
class Genome:
    pid: str
    kind: str                          # sensor|assoc|memory|meta|effect
    address: List[int]                 # 地址向量 ∈ {-1,0,1}^K（嗓音/听力）
    gain: float = 1.0                  # 场增益
    theta0: float = 0.6                # 初生阈值
    rho_star: float = 0.08             # 目标发放率
    parent: str = ""                   # 亲代 pid
    birth_tick: int = 0
    tool: Optional[str] = None         # effect：工具名或 macro:<名>
    recipe: Optional[List[str]] = None # 宏配方（步步可逆）
    mutations: int = 0                 # 变异次数（谱系痕迹）
    gen: int = 0                       # 第几代

    @staticmethod
    def random(pid: str, kind: str, rng: random.Random,
               birth_tick: int = 0, parent: str = "",
               tool: Optional[str] = None,
               recipe: Optional[List[str]] = None,
               gen: int = 0) -> "Genome":
        while True:
            address = [rng.choice((-1, 0, 1)) for _ in range(K_CHANNELS)]
            if any(address):
                break
        return Genome(pid=pid, kind=kind, address=address,
                      gain=rng.uniform(0.7, 1.4),
                      theta0=rng.uniform(0.30, 0.78),
                      rho_star=rng.uniform(0.04, 0.16),
                      parent=parent, birth_tick=birth_tick,
                      tool=tool, recipe=list(recipe) if recipe else None,
                      gen=gen)

    def copy_with(self, pid: str, birth_tick: int) -> "Genome":
        return Genome(pid=pid, kind=self.kind, address=list(self.address),
                      gain=self.gain, theta0=self.theta0,
                      rho_star=self.rho_star, parent=self.parent,
                      birth_tick=birth_tick, tool=self.tool,
                      recipe=list(self.recipe) if self.recipe else None,
                      mutations=self.mutations, gen=self.gen + 1)

    def mutate(self, rng: random.Random) -> None:
        """内在可塑性：有界的小步变异。"""
        self.mutations += 1
        c = rng.randrange(K_CHANNELS)
        self.address[c] = rng.choice((-1, 0, 1))
        if not any(self.address):                     # 保持嗓音非零
            self.address[rng.randrange(K_CHANNELS)] = rng.choice((-1, 1))
        self.gain = clamp(self.gain * rng.uniform(0.95, 1.05), GAIN_MIN, GAIN_MAX)
        self.rho_star = clamp(self.rho_star * rng.uniform(0.90, 1.10),
                              RHO_MIN, RHO_MAX)

    def to_dict(self) -> Dict:
        return {"pid": self.pid, "kind": self.kind,
                "address": list(self.address),
                "gain": round(self.gain, 4), "theta0": round(self.theta0, 4),
                "rho_star": round(self.rho_star, 4), "parent": self.parent,
                "birth_tick": self.birth_tick, "tool": self.tool,
                "recipe": list(self.recipe) if self.recipe else None,
                "mutations": self.mutations, "gen": self.gen}


# ----------------------------------------------------------------------
# 键：无权重的拓扑边（只决定相邻关系，绝不参与信号传播强度）
# ----------------------------------------------------------------------
@dataclass
class Bond:
    other: str
    cofire: float = 1.0                # 共燃计数（衰减制）
    created: int = 0


@dataclass
class ActionContext:
    """执行基元行动时由引擎装配的环境切片。"""
    env: object
    tick: int
    context_sig: str
    constraints: object
    confirm_token: Optional[str] = None   # 多层确认的人类令牌


# ----------------------------------------------------------------------
# 基元基类
# ----------------------------------------------------------------------
class Primitive:
    KIND = "assoc"

    def __init__(self, genome: Genome, rng: random.Random):
        self.genome = genome
        self.rng = rng
        self.p = 0.0                    # 势能
        self.theta = genome.theta0      # 发放阈值（自适应）
        self.e = INIT_ENERGY            # 能量账本
        self.burst = 0.0                # 当前爆发值
        self.refrac = 0                 # 不应期剩余
        self.rate_ema = genome.rho_star # 发放率 EMA
        self.pred = 0.0                 # 单步预测器（对最敏感通道）
        self.surprise_ema = 0.3         # 惊讶度 EMA
        self.comp_ema = 0.5             # 胜任度 EMA
        self.bonds: Dict[str, Bond] = {}
        self.lateral_in = 0.0           # 待吸收的侧抑制量
        self.pending_drive = 0.0        # 外部注入（教导脉冲 / 记忆重放）
        self.snapshot_drive = 0.0       # meta 专用：引擎推送的全局快照
        self.alive = True
        self.last_fire_tick = -999
        self.fires = 0                  # 一生发放总次数
        self.replays = 0                # 记忆重放次数

    # ---- 身份 ----
    @property
    def pid(self) -> str:
        return self.genome.pid

    @property
    def kind(self) -> str:
        return self.genome.kind

    def sensitive_channel(self) -> int:
        addr = self.genome.address
        return max(range(K_CHANNELS), key=lambda c: abs(addr[c]))

    def motto(self) -> str:
        return MOTTOS.get(self.kind, MOTTOS["assoc"])

    # ---- 感受与更新 ----
    def listen_drive(self, fld: List[float], obs_code: Optional[List[float]]) -> float:
        """输入 = 自身地址对场（及传感码）的无偏投影 × 增益。
        注意：均值聚合本身无偏（见 FieldMedium）；地址向量是基元自己的嗓音，
        属于其内在身份，不构成对他人贡献的加权。"""
        g = self.genome.gain
        d = 0.0
        for c in range(K_CHANNELS):
            a = self.genome.address[c]
            if a:
                d += a * fld[c]
                if obs_code is not None:
                    d += a * obs_code[c]
        nnz = sum(1 for a in self.genome.address if a)
        return g * d / math.sqrt(nnz)

    def update(self, drive: float) -> bool:
        """局部动力学一步。返回本拍是否发放。"""
        drive = clamp(drive + self.pending_drive
                      + self.rng.gauss(0.0, NOISE_STD), -6.0, 6.0)
        self.pending_drive = 0.0
        self.p = self.p * (1.0 - LAMBDA) + drive - self.lateral_in
        self.lateral_in = 0.0
        fired = False
        if self.refrac > 0:
            self.refrac -= 1
        elif self.p >= self.theta:
            fired = True
            self.burst = BURST0
            self.refrac = REFRAC
            self.p -= self.theta
            self.e -= SPIKE_COST
            self.fires += 1
        self.burst *= BURST_DECAY
        if self.burst < 0.01:
            self.burst = 0.0
        r = 1.0 if fired else 0.0
        self.rate_ema += RATE_ALPHA * (r - self.rate_ema)
        self.theta = clamp(self.theta + ETA_THETA * (self.rate_ema - self.genome.rho_star),
                           THETA_MIN, THETA_MAX)
        self.e -= UPKEEP + BOND_UPKEEP * len(self.bonds)
        self.e = clamp(self.e, -1.0, ENERGY_CAP)
        return fired

    def learn(self, fld: List[float]) -> None:
        """单步预测器 → 惊讶度 → 胜任度。"""
        c = self.sensitive_channel()
        actual = fld[c]
        err = abs(actual - self.pred)
        self.pred += PRED_ALPHA * (actual - self.pred)
        self.surprise_ema += 0.05 * (err - self.surprise_ema)
        comp = math.exp(-err / COMP_SIGMA)
        self.comp_ema += 0.02 * (comp - self.comp_ema)

    # ---- 对外交互 ----
    def emit_contribution(self) -> List[float]:
        b = self.burst * EMIT_SCALE   # 公共单位制：人人同倍，均值仍无偏
        return [b * a for a in self.genome.address]

    def receive_lateral(self, amount: float) -> None:
        self.lateral_in += amount

    def inject_pulse(self, strength: float) -> None:
        """外部教导脉冲进入下一拍的驱动项（有界）。"""
        self.pending_drive += clamp(strength, -2.0, 2.0)

    # ---- 结构 ----
    def add_bond(self, other_pid: str, tick: int) -> bool:
        if other_pid == self.pid or other_pid in self.bonds:
            return False
        if len(self.bonds) >= BOND_CAP:
            return False
        self.bonds[other_pid] = Bond(other=other_pid, cofire=TAU_BOND, created=tick)
        return True

    def remove_bond(self, other_pid: str) -> None:
        self.bonds.pop(other_pid, None)

    def cofire(self, other_pid: str) -> None:
        b = self.bonds.get(other_pid)
        if b is not None:
            b.cofire += 1.0

    def decay_bonds(self) -> List[str]:
        gone = []
        for pid, b in list(self.bonds.items()):
            b.cofire *= BOND_DECAY
            if b.cofire < BOND_PRUNE:
                gone.append(pid)
        for pid in gone:
            self.remove_bond(pid)
        return gone

    # ---- 谱系 ----
    def mitosis_child(self, new_pid: str, tick: int) -> "Primitive":
        child_g = self.genome.copy_with(new_pid, tick)
        child_g.mutate(self.rng)
        child = make_primitive(child_g, self.rng)
        half = self.e / 2.0
        self.e -= half
        child.e = half
        return child

    # ---- 自述 ----
    def describe(self, tick: int) -> str:
        g = self.genome
        age = tick - g.birth_tick
        nl = chr(10)
        lines = [
            "[" + g.pid + "] " + self.kind + " · 第" + str(g.gen) + "代 · 生于tick "
                + str(g.birth_tick) + "（年龄" + str(age) + "拍）",
            "  格言：" + self.motto(),
            "  能量 %.2f｜势 %.2f/阈 %.2f｜发放率EMA %.3f（目标 %.3f）" % (
                self.e, self.p, self.theta, self.rate_ema, g.rho_star),
            "  嗓音 " + "".join(str(a) if a >= 0 else "‑" for a in g.address)
                + "｜增益 %.2f｜变异 %d 次" % (g.gain, g.mutations),
            "  键 %d 条｜胜任度 %.2f｜一生发放 %d 次" % (
                len(self.bonds), self.comp_ema, self.fires),
        ]
        if g.tool:
            extra = "  工具：" + g.tool
            if g.recipe:
                extra += "（配方：" + "→".join(g.recipe) + "）"
            lines.append(extra)
        return nl.join(lines)


# ----------------------------------------------------------------------
# 五种分化型
# ----------------------------------------------------------------------
class SensorPrimitive(Primitive):
    """感知基元：世界的入口。listen 时额外接收环境编码向量。"""
    KIND = "sensor"


class AssocPrimitive(Primitive):
    """关联基元：纯内部耦合子，结构生长的主体材料。"""
    KIND = "assoc"


class MemoryPrimitive(Primitive):
    """记忆基元：维护敏感时期的场快照环形缓冲，寂静时低概率重放（巩固）。"""
    KIND = "memory"

    def __init__(self, genome: Genome, rng: random.Random):
        super().__init__(genome, rng)
        self.snapshots: deque = deque(maxlen=48)

    def learn(self, fld: List[float]) -> None:
        super().learn(fld)
        if self.burst > 0.4:
            self.snapshots.append(tuple(round(v, 3) for v in fld))

    def maybe_replay(self) -> bool:
        """寂静且低概率时重放一段旧场模式作为内部驱动。返回是否重放。"""
        if (self.snapshots and self.rate_ema < self.genome.rho_star * 0.8
                and self.rng.random() < REPLAY_P):
            snap = self.snapshots[self.rng.randrange(len(self.snapshots))]
            c = self.sensitive_channel()
            self.inject_pulse(clamp(snap[c], -1.5, 1.5))
            self.replays += 1
            return True
        return False


class MetaPrimitive(Primitive):
    """监察基元：接收引擎推送的全局量化快照，是自述报告的数据锚点。"""
    KIND = "meta"

    def listen_drive(self, fld, obs_code) -> float:
        return super().listen_drive(fld, obs_code) + clamp(self.snapshot_drive, -1.0, 1.0)


class EffectPrimitive(Primitive):
    """执行基元：携带工具名，以自身势能提出动议；原生安全门在其执行入口内。"""
    KIND = "effect"

    def motion_support(self) -> Optional[float]:
        """合格性初检（能量/不应期/持器）；情境约束否决由引擎裁定。"""
        if not self.genome.tool or self.e <= MIN_ACT_ENERGY or self.refrac > 0:
            return None
        return math.tanh(max(0.0, self.p))

    def execute(self, action: str, ctx: ActionContext) -> Dict:
        spec = ACTION_SPECS.get(action)
        if spec is None:
            if action.startswith("macro:"):
                spec = {"reversible": True, "desc": "宏工具"}
            else:
                return {"ok": False, "why": "unknown_action", "action": action}
        if not spec["reversible"] and ctx.confirm_token is None:
            raise IrreversibleBlocked(
                action + " 是不可逆动作：需要 监察法定票 + 人类令牌 + 回滚方案 三重确认")
        if ctx.constraints is not None and ctx.constraints.vetoed(action, ctx.context_sig):
            return {"ok": False, "why": "constraint_veto", "action": action,
                    "sig": ctx.context_sig}
        result = ctx.env.apply(action)
        return {"ok": True, "action": action, "result": result,
                "reversible": bool(spec["reversible"])}


_KIND_CLS = {"sensor": SensorPrimitive, "assoc": AssocPrimitive,
             "memory": MemoryPrimitive, "meta": MetaPrimitive,
             "effect": EffectPrimitive}


def make_primitive(genome: Genome, rng: random.Random) -> Primitive:
    return _KIND_CLS[genome.kind](genome, rng)


def macro_is_reversible(recipe: Optional[List[str]]) -> bool:
    """宏配方安全性由构成步骤决定：步步可逆 ⇒ 宏必可逆。"""
    if not recipe:
        return False
    return all(ACTION_SPECS.get(s, {}).get("reversible", False) for s in recipe)
