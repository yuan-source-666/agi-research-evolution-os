# -*- coding: utf-8 -*-
"""
evolution_engine.py —— 演化引擎：自然法则层（不是控制器）
==========================================================
本模块对全体基元一视同仁地施加五条自然规则，从不点名、从不排序、从不路由：

  法则1 赫布加强   —— 共燃者缔结拓扑键（无权重，只有相邻关系）
  法则2 能量竞争   —— 等额红利 + 有界开销；收不抵支者自然溶解（竞争抑制）
  法则3 自适应阈值 —— 每个基元把自身发放率拉向自己的目标率（稳态）
  法则4 内在可塑性 —— 与惊讶度成正比的有界自我变异
  法则5 侧向抑制   —— 发放者压制键邻居，逼出功能分化

另含：共享场（无偏均值聚合）、演示环境、法定人数表决、工具锻造炉、
原生安全的不可逆操作多层确认管线，以及成长记录（JSONL）。
纯标准库。
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from primitives import (
    ACTION_SPECS, K_CHANNELS, SPLIT_AT, THETA_MAX, THETA_MIN,
    COMP_SIGMA, MUT_P_BASE, IrreversibleBlocked, ActionContext, Genome,
    MemoryPrimitive, EffectorPrimitive, MetaPrimitive, SensorPrimitive,
    clamp, make_primitive,
)

REWARD_K = 15.0    # 可预测性改善 → 奖励的换算
BASE_R = 0.05      # 成功行动的保底奖励


# ----------------------------------------------------------------------
# 共享场：唯一的通信介质。field[c] = 全体贡献的算术平均（无偏）。
# ----------------------------------------------------------------------
class FieldMedium:
    def __init__(self, k: int = K_CHANNELS):
        self.k = k
        self.values: List[float] = [0.0] * k
        self.hist: deque = deque(maxlen=64)

    def compute_from(self, population: List) -> None:
        acc = [0.0] * self.k
        for pr in population:
            b = pr.burst
            if b == 0.0:
                continue
            a = pr.g.address
            for c in range(self.k):
                if a[c]:
                    acc[c] += b * a[c]
        n = max(1, len(population))          # 分母是全体存活基元数
        self.values = [x / n for x in acc]
        self.hist.append(list(self.values))


# ----------------------------------------------------------------------
# 演示环境：一维资源世界。中心是唯一的低湍区——恢复力微弱且缓慢，
# 湍流随偏离中心的距离二次增长。执行基元的"轻推"能快速把世界拉回
# 低湍区；世界越平静越可预测，而可预测性是群落的唯一收入来源。
# 于是"稳住世界"无需任何外部目标函数即会自发涌现。
# ----------------------------------------------------------------------
class ResourceEnv:
    def __init__(self, rng: random.Random, x: float = 50.0,
                 gradient: float = 0.25, noise_base: float = 0.8,
                 noise_turb: float = 2.5, delta: float = 12.0):
        self.rng = rng
        self.x = x
        self.gradient = gradient          # 向心恢复力强度
        self.noise_base = noise_base      # 基础湍流
        self.noise_turb = noise_turb      # 偏离湍流系数（×dist²）
        self.delta = delta                # 单次轻推位移
        self.t = 0
        self.hist: deque = deque(maxlen=64)
        self.hist.append(x)
        self.last_probe: Optional[float] = None

    def step(self) -> float:
        dist = abs(self.x - 50.0) / 50.0
        drift = -self.gradient * math.tanh((self.x - 50.0) / 25.0)
        drift += self.rng.gauss(0.0,
                                self.noise_base + self.noise_turb * dist * dist)
        self.x = min(100.0, max(0.0, self.x + drift))
        self.t += 1
        self.hist.append(self.x)
        self.last_probe = None
        return self.x

    def bucket01(self) -> float:
        return clamp(self.x / 100.0, 0.0, 1.0)

    def op(self, name: str) -> Dict:
        if name == "probe":
            self.last_probe = self.x
            return {"x": round(self.x, 2)}
        if name == "nudge_plus":
            dx = min(self.delta, max(0.0, 100.0 - self.x))
            self.x += dx
            return {"dx": round(dx, 2), "x": round(self.x, 2)}
        if name == "nudge_minus":
            dx = min(self.delta, self.x)
            self.x -= dx
            return {"dx": -round(dx, 2), "x": round(self.x, 2)}
        if name == "reset":
            self.x = 50.0                      # 不可逆：由原生安全门把关
            return {"reset": True, "x": 50.0}
        raise KeyError(name)

    def potential(self) -> float:
        """稳定势：距中心越近越高。动作结算用它做确定性归因。"""
        return math.exp(-((self.x - 50.0) / 30.0) ** 2)

    def predictability(self) -> float:
        xs = list(self.hist)
        if len(xs) < 8:
            return 0.5
        m = sum(xs) / len(xs)
        var = sum((v - m) ** 2 for v in xs) / len(xs)
        return math.exp(-var / 200.0)

    def clone(self, rng: Optional[random.Random] = None) -> "ResourceEnv":
        c = ResourceEnv(rng if rng is not None else self.rng,
                        x=self.x, gradient=self.gradient,
                        noise_base=self.noise_base,
                        noise_turb=self.noise_turb, delta=self.delta)
        c.t = self.t
        c.hist = deque(self.hist, maxlen=64)
        c.last_probe = self.last_probe
        return c


# ----------------------------------------------------------------------
# 约束库：人类纠正以"新的本地事实"入册，供合格性审查查询
# ----------------------------------------------------------------------
class ConstraintStore:
    def __init__(self):
        self.entries: List[Dict] = []

    def add(self, action: str, polarity: str, context_sig: str,
            note: str, tick: int) -> Dict:
        e = {"action": action, "polarity": polarity, "context": context_sig,
             "note": note, "tick": tick}
        self.entries.append(e)
        return e

    def blocked(self, action: str, context_sig: str) -> Optional[Dict]:
        for e in reversed(self.entries):
            if e["polarity"] == "negative" and e["action"] == action:
                if context_sig == e["context"] or e["context"] == "*":
                    return e
        return None

    def __len__(self) -> int:
        return len(self.entries)


# ----------------------------------------------------------------------
# 度量登记簿 + 成长日志
# ----------------------------------------------------------------------
class MetricsRegistry:
    NAMES = ("population", "bonds", "components", "mean_degree", "pool",
             "competence", "novelty", "coherence", "arousal")

    def __init__(self, cap: int = 600):
        self.series: Dict[str, deque] = {n: deque(maxlen=cap)
                                         for n in self.NAMES}
        self.counters: Counter = Counter()

    def record(self, **vals) -> None:
        for k, v in vals.items():
            if k in self.series:
                self.series[k].append(v)

    def last(self, name: str) -> Optional[float]:
        s = self.series[name]
        return s[-1] if s else None

    def spark(self, name: str, width: int = 28) -> str:
        s = list(self.series[name])[-width:]
        if len(s) < 2:
            return "—"
        lo, hi = min(s), max(s)
        rng = (hi - lo) or 1.0
        blocks = "▁▂▃▄▅▆▇█"
        return "".join(blocks[int((v - lo) / rng * 7.999)] for v in s)


class GrowthJournal:
    """growth_log.jsonl（量化快照）/ events.jsonl（离散事件）/ transcript.jsonl"""

    def __init__(self, outdir: str):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.tail: deque = deque(maxlen=12)

    def _append(self, fname: str, obj: Dict) -> None:
        with open(os.path.join(self.outdir, fname), "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def event(self, tick: int, etype: str, **kw) -> None:
        row = {"tick": tick, "type": etype}
        row.update(kw)
        self._append("events.jsonl", row)
        brief = ", ".join(f"{k}={v}" for k, v in kw.items())
        self.tail.appendleft(f"[t{tick}] {etype} {brief}")

    def metrics_row(self, tick: int, row: Dict) -> None:
        row = dict(row)
        row["tick"] = tick
        self._append("growth_log.jsonl", row)

    def dialogue(self, role: str, text: str) -> None:
        self._append("transcript.jsonl",
                     {"role": role, "text": text})

    def note(self, text: str) -> None:
        self.dialogue("system", text)


# ----------------------------------------------------------------------
# 会计：最近一次集体行动的因果链 + 历史操作序列（供解释与锻造引用）
# ----------------------------------------------------------------------
class Accountant:
    def __init__(self):
        self.last: Optional[Dict] = None
        self.history: deque = deque(maxlen=256)

    def record_action(self, chain: Dict) -> None:
        self.last = chain
        self.history.append(chain)

    def recipe_candidates(self, constraints: ConstraintStore,
                          ctx: str, limit: int = 3) -> List[str]:
        """最近成功且全程可逆的操作名，按新旧去重排序。"""
        out: List[str] = []
        for chain in reversed(self.history):
            if not chain.get("ok"):
                continue
            steps = list(chain.get("steps") or [])
            if not all(ACTION_SPECS.get(s, {}).get("reversible")
                       for s in steps):
                continue
            for s in steps:
                if s in out or constraints.blocked(s, ctx) is not None:
                    continue
                out.append(s)
        return out[:limit]


# ----------------------------------------------------------------------
# 工具锻造炉：监视反复得不到满足的动议签名，复合宏工具（步步可逆）
# ----------------------------------------------------------------------
class ToolForge:
    FORGE_COLD = 6      # 同一签名冷寂次数阈值
    FORGE_SIG  = 5      # 同一签名至少见过几次才算熟悉

    def __init__(self, engine: "SwarmEngine"):
        self.engine = engine
        self.sigs: Counter = Counter()
        self.cold: Counter = Counter()
        self.cooldown_until = 300       # 开局先让社会自己跑一会儿
        self.attempts = 0
        self.born: List[Dict] = []

    # ---- 信号采集 ----
    def observe_proposal(self, fam: str, ctx: str) -> None:
        key = (fam, ctx)
        self.sigs[key] += 1
        self.cold[key] += 1                       # 提了但没被选中

    def observe_selected(self, fam: str, ctx: str) -> None:
        self.sigs[(fam, ctx)] += 1

    def observe_quality(self, fam: str, ctx: str, quality: float) -> None:
        if quality > 0.02:
            self.cold[(fam, ctx)] = 0

    def observe_starve(self, ctx: str) -> None:
        key = ("*starve*", ctx)
        self.sigs[key] += 1
        self.cold[key] += 1

    # ---- 锻造 ----
    def maybe_forge(self, tick: int) -> Optional[Dict]:
        if tick < self.cooldown_until:
            return None
        cand = None
        for key, c in self.cold.items():
            if c >= self.FORGE_COLD and self.sigs.get(key, 0) >= self.FORGE_SIG:
                if cand is None or c > self.cold[cand]:
                    cand = key
        if cand is None:
            return None
        fam, ctx = cand
        eng = self.engine
        ops = eng.accountant.recipe_candidates(eng.constraints, ctx, limit=4)
        if len(ops) < 2:
            return None
        # 试错：由候选操作生成若干连续窗口配方，各自在克隆环境（独立
        # 随机源，不污染主宇宙）中干跑并模拟20拍，按稳定势增量评分。
        cands = []
        for lo in range(len(ops)):
            for hi in range(lo + 2, len(ops) + 1):
                cands.append(ops[lo:hi])
        cands = cands[:4]
        scored = []
        for rc in cands:
            env2 = eng.env.clone(random.Random(eng.rng.randrange(2 ** 31)))
            try:
                ok = all(ACTION_SPECS[s]["reversible"] for s in rc)
                if not ok:
                    continue
                for s in rc:
                    env2.op(s)
                for _ in range(20):
                    env2.step()
                scored.append((env2.potential(), rc))
            except Exception:
                continue
        if not scored:
            return None
        scored.sort(key=lambda sr: -sr[0])
        best_pot, recipe = scored[0]
        self.cold[cand] = 0
        self.cooldown_until = tick + 400
        self.attempts += 1
        name = f"宏{len(self.born) + 1}:{'+'.join(recipe)}"
        g = Genome.random(pid=eng._new_pid("E"), kind="effect",
                          rng=eng.rng, birth_tick=tick, parent="forge",
                          tool="macro:" + name, recipe=recipe)
        rec = {"name": name, "recipe": recipe, "tick": tick, "sig": fam,
               "score": round(best_pot, 4), "tried": len(scored)}
        self.born.append(rec)
        return {"genome": g, **rec}


# ----------------------------------------------------------------------
# 词汇册（语义萌芽）：教导时把一个词绑定到当下的场模式质心；
# 此后报告可诚实地说「当前情境接近你教我的某词（相似度 x）」。
# 词不是预装的，是经验中被命名的——这就是词扎根的最小实现。
# ----------------------------------------------------------------------
class Lexicon:
    SIM_THRESHOLD = 0.25     # 低于此相似度不攀亲戚

    def __init__(self):
        self.entries: List[Dict] = []

    @staticmethod
    def extract_word(text: str) -> str:
        m = re.search("\u300c(.+?)\u300d", text)                      # 「X」
        if not m:
            m = re.search("\u0022(.+?)\u0022", text)                  # "X"
        if m:
            return m.group(1).strip()[:12]
        return text.strip()[:12]

    def bind(self, word: str, pattern: List[float], tick: int) -> Dict:
        for e in self.entries:
            if e["word"] == word:
                # 重教即巩固：模式向新经验缓慢滑动
                e["pattern"] = [0.7 * a + 0.3 * b
                                for a, b in zip(e["pattern"], pattern)]
                e["uses"] += 1
                e["last_tick"] = tick
                return e
        e = {"word": word, "pattern": list(pattern),
             "bound_tick": tick, "last_tick": tick, "uses": 1}
        self.entries.append(e)
        return e

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(x * x for x in b)) or 1e-9
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    def nearest(self, pattern: List[float]) -> Optional[Dict]:
        best, best_sim = None, self.SIM_THRESHOLD
        for e in self.entries:
            sim = self._cos(pattern, e["pattern"])
            if sim > best_sim:
                best, best_sim = e, sim
        if best is None:
            return None
        return {"word": best["word"], "sim": round(best_sim, 3)}

    def to_list(self, limit: int = 12) -> List[Dict]:
        return [{"word": e["word"], "bound_tick": e["bound_tick"],
                 "uses": e["uses"]} for e in self.entries[:limit]]

    def __len__(self) -> int:
        return len(self.entries)


# ----------------------------------------------------------------------
# 引擎配置
# ----------------------------------------------------------------------
@dataclass
class EngineConfig:
    seed: int = 42
    n0: int = 64              # 初始种群
    cap: int = 256            # 种群硬上限
    q_every: int = 8          # 表决周期（拍）
    theta_act: float = 0.10   # 集体唤醒触发线（近期活动率）
    dividend: float = 0.08    # 每拍等额红利上限
    maint: float = 0.010      # 维持能耗
    bond_upkeep: float = 0.001
    yield_k: float = 3.0      # 可预测性收益系数
    tau_bond: int = 4         # 缔键共燃阈值
    bond_window: int = 2
    eta_theta: float = 0.05
    inhib: float = 0.15       # 侧向抑制强度
    outdir: str = "growth"
    meta_every: int = 20
    journal_every: int = 25


_KIND_TAG = {"sensor": "S", "assoc": "A", "memory": "M",
             "meta": "Y", "effect": "E"}


# ----------------------------------------------------------------------
# 群落引擎
# ----------------------------------------------------------------------
class SwarmEngine:
    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()
        self.rng = random.Random(self.cfg.seed)
        self.env = ResourceEnv(self.rng)
        self.fields = FieldMedium()
        self.metrics = MetricsRegistry()
        self.journal = GrowthJournal(self.cfg.outdir)
        self.constraints = ConstraintStore()
        self.accountant = Accountant()
        self.forge = ToolForge(self)
        self.population: List = []
        self.by_id: Dict[str, object] = {}
        self.pair_cofire: Counter = Counter()
        self.recent_spikes: deque = deque(maxlen=128)   # (tick, pid)
        self.teachings: List[Dict] = []
        self.lexicon = Lexicon()
        self.lineage: Dict[str, int] = {}      # pid → 世系深度
        self.pending_irreversible: Optional[Dict] = None
        self.pool = 50.0            # 初始能量池拨给
        self.paused = False
        self.t = 0
        self._next_pid = 0
        self._sync_ema = 0.0
        self._act = 0.0
        self._build_initial()

    # ---------- 构造 ----------
    def _new_pid(self, tag: str) -> str:
        self._next_pid += 1
        return f"{tag}{self._next_pid:04d}"

    def _activity(self) -> float:
        """集体唤醒度：最近4拍内至少发放过一次的基元占比。"""
        if not self.population:
            return 0.0
        lo = self.t - 3
        active = set()
        for tk, pid in reversed(self.recent_spikes):
            if tk < lo:
                break
            active.add(pid)
        return len(active) / len(self.population)

    def _admit_genome(self, g: Genome, depth: int = 0) -> object:
        p = make_primitive(g, self.rng)
        self.population.append(p)
        self.by_id[g.pid] = p
        self.lineage[g.pid] = depth
        return p

    def _build_initial(self) -> None:
        plan = [("sensor", 12), ("memory", 8), ("meta", 6)]
        tools = ["probe"] * 3 + ["nudge_plus"] * 4 + \
                ["nudge_minus"] * 4 + ["reset"] * 2
        for kind, n in plan:
            for _ in range(n):
                g = Genome.random(self._new_pid(_KIND_TAG[kind]), kind,
                                  self.rng)
                self._admit_genome(g)
        for tl in tools:
            g = Genome.random(self._new_pid("E"), "effect", self.rng, tool=tl)
            self._admit_genome(g)
        while len(self.population) < self.cfg.n0:
            g = Genome.random(self._new_pid("A"), "assoc", self.rng)
            self._admit_genome(g)
        self.journal.event(0, "GENESIS",
                           population=len(self.population),
                           kinds=dict(Counter(p.KIND for p in self.population)))

    # ---------- 主循环 ----------
    def step(self) -> None:
        cfg = self.cfg
        self.t += 1
        t = self.t

        # -- 0 凤凰条款：彻底灭绝且池有储备时，重新萌发（记录在案）--
        if not self.population and self.pool >= 20.0:
            self.pool -= 20.0
            plan = [("sensor", 4), ("memory", 2), ("meta", 2),
                    ("effect", 4), ("assoc", 8)]
            tools = ["probe", "nudge_plus", "nudge_minus"]
            ti = 0
            for kind, n in plan:
                for _ in range(n):
                    g = Genome.random(self._new_pid(_KIND_TAG[kind]), kind,
                                      self.rng, birth_tick=t)
                    if kind == "effect":
                        g.tool = tools[ti % len(tools)]
                        ti += 1
                    self._admit_genome(g)
            self.journal.event(t, "PHOENIX", population=len(self.population),
                               pool=round(self.pool, 2))

        # -- 1 环境 --
        x = self.env.step()
        sensors = [p for p in self.population
                   if isinstance(p, SensorPrimitive)]
        coarse = self.env.bucket01()
        for s in sensors:
            s.observe_coarse(coarse)
        if self.env.last_probe is not None:
            fine = clamp(self.env.last_probe / 100.0, 0.0, 1.0)
            for s in sensors:
                s.observe_fine(fine, strength=0.8)

        # -- 2 记忆：记录上一拍场；偶发重放（巩固） --
        for p in self.population:
            if isinstance(p, MemoryPrimitive):
                p.remember(self.fields.values)
                p.maybe_replay()

        # -- 3 场：无偏均值聚合 --
        self.fields.compute_from(self.population)

        # -- 4 局部动力学 --
        fired = []
        fld = self.fields.values
        for p in self.population:
            p.predict(fld)
            if p.update(p.drive_from(fld), t):
                fired.append(p)
        pop = max(1, len(self.population))
        self._sync_ema += 0.05 * (len(fired) / pop - self._sync_ema)

        # -- 5 五条自然法则 --
        self._law_hebbian(fired, t)
        self._law_lateral(fired)
        self._law_homeostasis()
        self._law_economy()
        self._law_plasticity(fired, t)

        # -- 6 生老病死 --
        self._lifecycle(t)

        # -- 7 集体决策 --
        self._act = self._activity()
        if self.population and (t % cfg.q_every == 0
                                or self._act >= cfg.theta_act * 2.0):
            self._collective_decision(t, self._act)

        # -- 8 工具锻造 --
        forged = self.forge.maybe_forge(t)
        if forged is not None:
            self._admit_genome(forged["genome"])
            self.metrics.counters["tools_born"] += 1
            self.journal.event(t, "TOOL_BORN", name=forged["name"],
                               recipe="+".join(forged["recipe"]),
                               sig=forged["sig"])

        # -- 9 监察快照 --
        if t % cfg.meta_every == 0:
            snap = self.snapshot()
            for p in self.population:
                if isinstance(p, MetaPrimitive):
                    p.push_snapshot(snap)

        # -- 10 度量与成长日志 --
        if t % cfg.journal_every == 0:
            self._record_metrics(t)

    def run(self, ticks: int, stop=None, on_report=None) -> None:
        for _ in range(int(ticks)):
            if stop is not None and stop():
                break
            while self.paused:
                if stop is not None and stop():
                    return
                time.sleep(0.05)
            self.step()
            if on_report is not None and self.t % self.cfg.journal_every == 0:
                on_report(self)

    # ---------- 法则 ----------
    def _law_hebbian(self, fired: List, t: int) -> None:
        cfg = self.cfg
        # 先依据此前窗口内的共燃计数缔键（避免同拍双重计数）
        for f in fired:
            for tk, opid in reversed(self.recent_spikes):
                if tk < t - cfg.bond_window:
                    break
                if opid == f.g.pid:
                    continue
                key = (opid, f.g.pid) if opid < f.g.pid else (f.g.pid, opid)
                self.pair_cofire[key] += 1
        for f in fired:
            self.recent_spikes.append((t, f.g.pid))
        # 缔结新键；久不共燃的键自然衰朽（竞争抑制的拓扑表达）
        for key in list(self.pair_cofire.keys()):
            a, b = key
            pa, pb = self.by_id.get(a), self.by_id.get(b)
            if pa is None or pb is None:
                del self.pair_cofire[key]      # 一方已溶解
                continue
            c = self.pair_cofire[key]
            bonded = (b in pa.bonds) or (a in pb.bonds)
            if not bonded and c >= cfg.tau_bond:
                pa.add_bond(b, t)
                pb.add_bond(a, t)
                pa.energy -= 0.01
                pb.energy -= 0.01
                c = float(cfg.tau_bond) - 1.0   # 缔结后续命值
                self.journal.event(t, "BOND+", pair=f"{a}+{b}")
            else:
                c *= 0.99
                if bonded and c < 0.6:
                    pa.drop_bond(b)
                    pb.drop_bond(a)
                    self.journal.event(t, "BOND-",
                                       pair=f"{a}+{b}", cofire=round(c, 2))
            self.pair_cofire[key] = c
            if c < 0.05:
                del self.pair_cofire[key]

    def _law_lateral(self, fired: List) -> None:
        inh = self.cfg.inhib
        for f in fired:
            for oid in list(f.bonds.keys()):
                o = self.by_id.get(oid)
                if o is not None:
                    o.inh = min(o.inh + inh, 0.6)

    def _law_homeostasis(self) -> None:
        eta = self.cfg.eta_theta
        decay = 1.0 - 0.05
        for p in self.population:
            p.rate *= decay
            p.theta = clamp(p.theta + eta * (p.rate - p.g.rho_star),
                            THETA_MIN, THETA_MAX)

    def _law_economy(self) -> None:
        cfg = self.cfg
        pop = len(self.population)
        if pop == 0:
            return
        income = self.env.predictability() * cfg.yield_k
        self.pool += income
        share = min(cfg.dividend, 0.8 * self.pool / pop)
        self.pool -= share * pop
        for p in self.population:
            p.energy -= cfg.maint + cfg.bond_upkeep * len(p.bonds)
            p.energy += share

    def _law_plasticity(self, fired: List, t: int) -> None:
        for f in fired:
            surprise = f.last_err / COMP_SIGMA
            if self.rng.random() < MUT_P_BASE * (0.5 + surprise):
                f.g.mutate(self.rng)
                f.ch = max(range(K_CHANNELS), key=lambda i: abs(f.g.address[i]))
                self.metrics.counters["mutations"] += 1
                if self.rng.random() < 0.02:
                    self.journal.event(t, "MUT", who=f.g.pid, kind=f.KIND)

    # ---------- 生命周期 ----------
    def _lifecycle(self, t: int) -> None:
        dead = [p for p in self.population if p.energy <= 0.0]
        for d in dead:
            self.population.remove(d)
            self.by_id.pop(d.g.pid, None)
            for o in self.population:
                o.drop_bond(d.g.pid)
            self.metrics.counters["deaths"] += 1
            self.journal.event(t, "DIE", who=d.g.pid, kind=d.KIND, age=d.age)
        if not dead and len(self.pair_cofire) > 4096:
            self.pair_cofire.clear()          # 防止计数表无界膨胀
        for p in list(self.population):
            if len(self.population) >= self.cfg.cap:
                break
            if p.energy >= SPLIT_AT:
                child = p.split(self._new_pid(_KIND_TAG[p.KIND]), t)
                self.population.append(child)
                self.by_id[child.g.pid] = child
                self.lineage[child.g.pid] = self.lineage.get(p.g.pid, 0) + 1
                self.metrics.counters["births"] += 1
                self.journal.event(t, "BIRTH", who=child.g.pid,
                                   parent=p.g.pid, kind=p.KIND)

    # ---------- 集体决策（法定人数表决） ----------
    def _collective_decision(self, t: int, arousal: float) -> None:
        cfg = self.cfg
        ctx = f"zone{int(clamp(self.env.x, 0.0, 99.99) // 20)}"
        cost = 0.05
        actx = ActionContext(env=self.env, tick=t, context_sig=ctx,
                             constraints=self.constraints)
        props = []
        for p in self.population:
            if isinstance(p, EffectorPrimitive) and p.eligible(actx, cost):
                s = p.propose()
                if s > 0.15:
                    props.append((s, p))
        if arousal < cfg.theta_act:
            return
        if not props:
            self.forge.observe_starve(ctx)
            self.journal.event(t, "STARVE", ctx=ctx,
                               arousal=round(arousal, 3))
            return
        props.sort(key=lambda sv: -sv[0])
        top = props[0][0]
        ties = [p for s, p in props if s >= top - 1e-9]
        pick = ties[0] if len(ties) == 1 else self.rng.choice(ties)
        second = next((s for s, p in props if p is not pick), 0.0)
        margin = top - second

        before = self.env.potential()
        try:
            result = pick.execute(actx)
        except IrreversibleBlocked as ex:
            self.journal.event(t, "DENIED", why=str(ex))
            result = {"ok": False, "actor": pick.g.pid, "steps": []}
        after = self.env.potential()
        quality = after - before          # 稳定势增量：推向低湍区为正
        ok = bool(result.get("ok"))
        # 推向低湍区得正奖励；反向动作付出代价（下限 -0.10）
        reward = max(BASE_R + REWARD_K * quality, -0.10) if ok else 0.0
        coalition = self._coalition_of(pick, t)
        if reward > 0 and coalition:
            share = reward / len(coalition)
            for cid in coalition:
                c = self.by_id.get(cid)
                if c is not None:
                    c.receive_reward(share)
        chain = {
            "tick": t, "actor": pick.g.pid, "tool": pick.g.tool,
            "steps": [r.get("op") for r in result.get("steps", [])],
            "support": round(top, 3), "margin": round(margin, 3),
            "arousal": round(arousal, 3), "quality": round(quality, 4),
            "reward": round(reward, 4), "coalition": coalition,
            "context": ctx, "ok": ok,
        }
        self.accountant.record_action(chain)
        self.metrics.counters["actions"] += 1
        self.journal.event(t, "ACTION", tool=pick.g.tool, ctx=ctx,
                           support=round(top, 3),
                           quality=round(quality, 4), ok=ok)
        for s, p in props:
            if p is not pick:
                self.forge.observe_proposal(p.family(), ctx)
        self.forge.observe_selected(pick.family(), ctx)
        self.forge.observe_quality(pick.family(), ctx, quality)

    def _coalition_of(self, actor, t: int) -> List[str]:
        ids = [actor.g.pid]
        for tk, pid in reversed(self.recent_spikes):
            if tk < t - 6 or len(ids) >= 12:
                break
            if pid != actor.g.pid and pid not in ids and pid in self.by_id:
                ids.append(pid)
        return ids

    # ---------- 结构统计 ----------
    def _components(self) -> int:
        parent = {p.g.pid: p.g.pid for p in self.population}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for p in self.population:
            for oid in p.bonds:
                if oid in parent:
                    ra, rb = find(p.g.pid), find(oid)
                    if ra != rb:
                        parent[ra] = rb
        return len({find(x) for x in parent})

    def snapshot(self) -> Dict:
        pop = len(self.population)
        byk = Counter(p.KIND for p in self.population)
        bonds = sum(len(p.bonds) for p in self.population) // 2
        comp = sum(p.comp_ema for p in self.population) / max(1, pop)
        novel = (sum(1 for p in self.population if p.age < 200) / pop
                 if pop else 0.0)
        arousal = self._act
        depths = [self.lineage.get(p.g.pid, 0) for p in self.population]
        lin_max = max(depths) if depths else 0
        lin_avg = (sum(depths) / len(depths)) if depths else 0.0
        return {
            "tick": self.t, "population": pop, "by_kind": dict(byk),
            "bonds": bonds, "components": self._components(),
            "mean_degree": round(bonds * 2 / max(1, pop), 3),
            "pool": round(self.pool, 3),
            "competence": round(comp, 4), "novelty": round(novel, 3),
            "coherence": round(self._sync_ema, 4),
            "arousal": round(arousal, 4),
            "env_x": round(self.env.x, 2),
            "predictability": round(self.env.predictability(), 4),
            "lineage_max": lin_max,
            "lineage_avg": round(lin_avg, 2),
        }

    def _record_metrics(self, t: int) -> None:
        snap = self.snapshot()
        c = self.metrics.counters
        self.metrics.record(
            population=snap["population"], bonds=snap["bonds"],
            components=snap["components"], mean_degree=snap["mean_degree"],
            pool=snap["pool"], competence=snap["competence"],
            novelty=snap["novelty"], coherence=snap["coherence"],
            arousal=snap["arousal"])
        row = dict(snap)
        row.update(by_kind=snap["by_kind"], births=c["births"],
                   deaths=c["deaths"], actions=c["actions"],
                   tools_born=c["tools_born"], mutations=c["mutations"])
        self.journal.metrics_row(t, row)

    # ---------- 人机接口（由通信层调用） ----------
    def apply_correction(self, target: str, positive: bool,
                         note: str = "") -> Dict:
        last = self.accountant.last

        def fam(tool):
            return (tool or "").split(":")[0]

        affected, matched = [], False
        if last and (target in ("any", "*", "")
                     or fam(last["tool"]) == target
                     or last["tool"] == target):
            matched = True
            affected = list(last["coalition"])
        if positive:
            for pid in affected:
                p = self.by_id.get(pid)
                if p is not None:
                    p.theta = clamp(p.theta - 0.10, THETA_MIN, THETA_MAX)
                    p.receive_reward(0.5)
            summary = {"matched": matched, "affected": affected,
                       "delta_theta": -0.10, "bonus": 0.5}
        else:
            ctx = last["context"] if matched else "*"
            entry = self.constraints.add(target, "negative", ctx, note, self.t)
            for pid in affected:
                p = self.by_id.get(pid)
                if p is not None:
                    p.theta = clamp(p.theta + 0.15, THETA_MIN, THETA_MAX)
                    p.receive_reward(-0.2)
            summary = {"matched": matched, "affected": affected,
                       "delta_theta": 0.15, "constraint": entry,
                       "context": ctx}
        self.metrics.counters["corrections"] += 1
        self.journal.event(self.t, "CORRECT",
                           target=target, polarity="+" if positive else "-",
                           matched=matched, n=len(affected), note=note[:60])
        return summary

    def teach(self, text: str) -> Dict:
        self.teachings.append({"tick": self.t, "text": text})
        mems = [p for p in self.population if isinstance(p, MemoryPrimitive)]
        for m in mems:
            m.receive_reward(0.10)
        # 语义萌芽：把词绑定到此刻的场模式（群体当下共同经验）
        word = Lexicon.extract_word(text)
        entry = None
        if word:
            entry = self.lexicon.bind(word, list(self.fields.values), self.t)
            self.journal.event(self.t, "WORD_BOUND", word=word,
                               uses=entry["uses"])
        self.journal.event(self.t, "TEACH", n_mems=len(mems),
                           text=text[:60])
        return {"stored": len(self.teachings), "memories_touched": len(mems),
                "word": word if entry else None}

    # ---- 不可逆操作的多层确认管线 ----
    def irreversible_request(self, tool: str) -> Dict:
        spec = ACTION_SPECS.get(tool)
        if spec is None:
            return {"ok": False, "why": f"未登记工具 {tool}"}
        if spec["reversible"]:
            return {"ok": False, "why": f"{tool} 是可逆操作，无需确认管线"}
        metas = [p for p in self.population if isinstance(p, MetaPrimitive)]
        awake = ([m for m in metas if m.p >= m.theta])
        ratio = len(awake) / max(1, len(metas))
        meta_ok = len(metas) >= 3 and ratio >= 2.0 / 3.0
        self.pending_irreversible = {"tool": tool, "meta_ok": meta_ok,
                                     "tick": self.t}
        self.journal.event(self.t, "IRREV_REQ", tool=tool,
                           meta_ratio=round(ratio, 2), meta_ok=meta_ok)
        if not meta_ok:
            return {"ok": False, "stage": "awaiting_meta",
                    "meta_awake_ratio": round(ratio, 2),
                    "why": "监察基元法定人数未清醒（需≥2/3），请求挂起"}
        return {"ok": True, "stage": "awaiting_human",
                "meta_ratio": round(ratio, 2),
                "why": "监察基元已过法定线，等待人类确认令牌"}

    def irreversible_confirm(self, token: str) -> Dict:
        pend = self.pending_irreversible
        if not pend or not pend.get("meta_ok"):
            return {"ok": False, "why": "没有通过监察法定线的待确认请求"}
        if not token or not isinstance(token, str) or len(token) < 4:
            return {"ok": False, "why": "人类令牌无效（需≥4字符）"}
        tool = pend["tool"]
        executor = None
        ctx = f"zone{int(clamp(self.env.x, 0.0, 99.99) // 20)}"
        actx = ActionContext(env=self.env, tick=self.t, context_sig=ctx,
                             constraints=self.constraints,
                             confirm_token=token)
        for p in self.population:
            if (isinstance(p, EffectorPrimitive) and p.family() == tool
                    and p.eligible(actx, 0.05)):
                executor = p
                break
        if executor is None:
            self.pending_irreversible = None
            return {"ok": False, "why": "该动作已被负校正约束封锁或无合格执行者"}
        result = executor.execute(actx)
        self.pending_irreversible = None
        self.metrics.counters["actions"] += 1
        self.journal.event(self.t, "IRREV_DONE", tool=tool,
                           actor=executor.g.pid,
                           ok=bool(result.get("ok")))
        return {"ok": bool(result.get("ok")), "actor": executor.g.pid,
                "steps": result.get("steps", [])}

    # ---------- 自省（供报告渲染的事实来源） ----------
    def introspect(self) -> Dict:
        snap = self.snapshot()
        c = self.metrics.counters
        j = self.journal
        tools = [{"name": k, "desc": v["desc"],
                  "reversible": v["reversible"]} for k, v in
                 ACTION_SPECS.items()]
        macros = [{"name": b["name"], "recipe": "+".join(b["recipe"]),
                   "born_tick": b["tick"]} for b in self.forge.born]
        return {
            **snap,
            "births_total": c["births"], "deaths_total": c["deaths"],
            "actions_total": c["actions"], "corrections_total": c["corrections"],
            "tools_born_total": c["tools_born"],
            "mutations_total": c["mutations"],
            "constraints_n": len(self.constraints),
            "teachings_n": len(self.teachings),
            "lexicon_n": len(self.lexicon),
            "lexicon": self.lexicon.to_list(),
            "lexicon_nearest": self.lexicon.nearest(list(self.fields.values)),
            "forge_attempts": self.forge.attempts,
            "macros": macros, "tools": tools,
            "cold_sigs": sum(1 for v in self.forge.cold.values() if v >= 3),
            "last_action": self.accountant.last,
            "recent_events": list(j.tail)[:8],
            "sparks": {n: self.metrics.spark(n) for n in
                       ("population", "pool", "competence")},
        }

    def close(self) -> None:
        self.journal.note(f"[t{self.t}] 会话结束。种群={len(self.population)} "
                          f"池={self.pool:.2f}")
