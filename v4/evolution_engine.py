# -*- coding: utf-8 -*-
"""
evolution_engine.py —— PRIMORDIA v4 · 演化引擎（自然法则层，非控制器）
=====================================================================
引擎不是中枢：它像物理定律一样，对全体基元一视同仁地施加同一条规则，
从不点名、从不排序、从不路由。结构变化完全由五条内在法则驱动：

  法则1 赫伯加强   —— 共燃缔键（无权重拓扑边）
  法则2 能量竞争   —— 可预测性收益均分红利；收不抵支者自然溶解
  法则3 自适应阈值 —— 每基元把自身发放率拉向自己的目标率
  法则4 内在可塑性 —— 惊讶驱动的有界基因组变异
  法则5 侧向抑制   —— 发放者压制键邻居，逼迫功能分化

外加三种结构事件：有丝分裂 / 溶解死亡 / 工具锻造（ToolForge）。

信号传播的唯一通道是 FieldMedium 的无偏算术均值：
      field[c] = ( Σᵢ burstᵢ·addressᵢ[c] ) / N      （N = 全体存活基元数）
沉默者的 0 与喧哗者的 1 以同一分母进入平均——没有加权、路由、注意力、优先级。

纯 Python 标准库实现。
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, deque
from typing import Dict, List, Optional

from primitives import (
    K_CHANNELS, ACTION_SPECS, Genome, Bond,
    IrreversibleBlocked, ActionContext, Primitive, make_primitive,
    macro_is_reversible, clamp, INIT_ENERGY, SPLIT_AT, ENERGY_CAP, POP_CAP,
    MIN_POP, MUT_P_BASE, TAU_BOND, BOND_CAP, LAT_INH, MIN_ACT_ENERGY,
)
CROWD_FREE = 6              # 免拥挤税的键数额度
CROWD_TAX = 0.0004          # 超额键每拍维护税（逼向模块化，抑制无脑抱团）

# ----------------------------------------------------------------------
# 引擎级常数
# ----------------------------------------------------------------------
QUORUM_PERIOD = 40          # 法定表决周期（拍）
AROUSAL_TRIGGER = 0.35      # 集体唤醒度触发线（实测波峰可达，双通道决策）
MOTION_MIN = 0.08           # 动议最低支持度
INCOME_BASE = 2.50          # 环境基础收益
REWARD_SCALE = 6.0          # 集体行动改善奖励尺度
FORGE_FAIL_NEED = 3         # 动议冷寂次数阈值
SEQ_NEED = 3                # 复合序列最少出现次数
FORGE_CHECK_EVERY = 20
SNAPSHOT_EVERY = 25         # 成长日志采样周期

PRAISE_ENERGY, PRAISE_THETA = 1.5, 0.06
FROWN_FINE, FROWN_THETA = 1.0, 0.10

_FREQS = (1.0, 1.7, 2.3, 3.1, 4.2, 5.3, 6.1, 7.7)


# ======================================================================
# 环境：一个"可预测性即资源"的小世界
# ======================================================================
class ResourceEnv:
    """隐藏标量状态 x 缓慢漂移；世界越可预测、x 越贴近漂移目标，
    收益越高。执行基元的轻推可以把 x 拉向目标带——这是社会唯一的
    "谋生手段"，也是集体学习的压力来源。"""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.x = rng.uniform(-0.5, 0.5)
        self.target = rng.uniform(-0.4, 0.4)
        self.band = 0.25
        self.t = 0
        self.hist = deque([0.02] * 32, maxlen=32)
        self.last_income = 0.0

    # ---- 演化 ----
    def tick(self) -> float:
        self.t += 1
        prev = self.x
        self.target = clamp(self.target + 0.03 * math.sin(self.t / 37.0)
                            + self.rng.gauss(0, 0.012), -0.8, 0.8)
        self.x = clamp(self.x + self.rng.gauss(0, 0.02), -1.0, 1.0)
        self.hist.append(abs(self.x - prev))
        self.last_income = self.income()
        return self.last_income

    def stability(self) -> float:
        m = sum(self.hist) / len(self.hist)
        return math.exp(-8.0 * m)

    def prox(self) -> float:
        return math.exp(-((self.x - self.target) / 0.30) ** 2)

    def in_band(self) -> bool:
        return abs(self.x - self.target) <= self.band

    def income(self) -> float:
        return INCOME_BASE * (0.35 + 0.65 * self.stability()) * (0.45 + 0.55 * self.prox())

    # ---- 感官 ----
    def observe(self) -> float:
        return clamp(self.x + self.rng.gauss(0, 0.02), -1.0, 1.0)

    def encode(self, obs: float) -> List[float]:
        return [math.tanh(1.3 * math.sin(f * obs)) for f in _FREQS]

    def context_sig(self) -> str:
        return ("P" if self.x >= self.target else "N") + ("b" if self.in_band() else "o")

    # ---- 行动 ----
    def clone(self) -> "ResourceEnv":
        e = ResourceEnv(random.Random(0))
        e.x, e.target, e.band, e.t = self.x, self.target, self.band, self.t
        e.hist = deque(self.hist, maxlen=32)
        return e

    def apply(self, action: str) -> Dict:
        if action == "probe":
            return {"probed_x": round(self.x, 3)}
        if action == "nudge_plus":
            self.x = clamp(self.x + 0.06, -1.0, 1.0)
            return {"nudged": +0.06}
        if action == "nudge_minus":
            self.x = clamp(self.x - 0.06, -1.0, 1.0)
            return {"nudged": -0.06}
        if action == "reset":
            self.x = 0.0                      # 不可逆；入口处已有多重确认门
            return {"reset": True}
        raise ValueError("unknown action: " + str(action))

    def state_dict(self) -> Dict:
        return {"x": round(self.x, 4), "target": round(self.target, 4),
                "band": self.band, "t": self.t, "hist": [round(v, 4) for v in self.hist]}


# ======================================================================
# 共享场介质：无偏均值聚合（全架构最强的约束）
# ======================================================================
class FieldMedium:
    def __init__(self):
        self.field: List[float] = [0.0] * K_CHANNELS
        self._acc: List[float] = [0.0] * K_CHANNELS
        self.n = 1

    def begin(self, n_alive: int) -> None:
        self._acc = [0.0] * K_CHANNELS
        self.n = max(1, n_alive)

    def submit(self, contribution: List[float]) -> None:
        for c in range(K_CHANNELS):
            self._acc[c] += contribution[c]

    def aggregate(self) -> List[float]:
        self.field = [self._acc[c] / self.n for c in range(K_CHANNELS)]
        return self.field


# ======================================================================
# 原生安全的记忆面：人类纠正形成的约束册
# ======================================================================
class ConstraintStore:
    def __init__(self, path: str):
        self.path = path
        self.map: Dict[str, Dict] = {}
        self.load()

    @staticmethod
    def _key(family: str, sig: str) -> str:
        return family + "|" + sig

    def add(self, family: str, sig: str, reason: str, tick: int) -> None:
        k = self._key(family, sig)
        rec = self.map.setdefault(k, {"count": 0, "reason": "", "tick": tick})
        rec["count"] += 1
        if reason:
            rec["reason"] = reason[:60]
        rec["tick"] = tick
        self.save()

    def relax(self, family: str, sig: str) -> bool:
        changed = False
        for s in (sig, "*"):
            k = self._key(family, s)
            if k in self.map and self.map[k]["count"] > 0:
                self.map[k]["count"] -= 1
                changed = True
        if changed:
            self.save()
        return changed

    def vetoed(self, family: str, sig: str) -> bool:
        for s in (sig, "*"):
            k = self._key(family, s)
            if k in self.map and self.map[k]["count"] > 0:
                return True
        return False

    def count(self) -> int:
        return sum(1 for v in self.map.values() if v["count"] > 0)

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.map, f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.map = json.load(f)
            except Exception:
                self.map = {}


# ======================================================================
# 问责窗：最近一次集体行动的完整记录（纠正的落点）
# ======================================================================
class AccountabilityWindow:
    def __init__(self):
        self.last: Optional[Dict] = None

    def record(self, tick: int, family: str, participants: List[str],
               winner: str, sig: str, delta: float) -> None:
        self.last = {"tick": tick, "family": family,
                     "participants": list(participants), "winner": winner,
                     "sig": sig, "delta": round(delta, 4)}


# ======================================================================
# 工具锻造炉：反复得不到支持的动议 ⇒ 复合历史成功序列 ⇒ 新宏工具基元
# ======================================================================
class ToolForge:
    def __init__(self):
        self.fail_counts: Counter = Counter()
        self.history: deque = deque(maxlen=240)
        self.recipes: Dict[str, List[str]] = {}
        self.forged = 0

    def record_fail(self, family: str, sig: str = "") -> None:
        # 冷寂按动作族记账：同一工具的需求不因情境不同而被拆散
        self.fail_counts[family] += 1

    def record_success(self, actions: List[str]) -> None:
        self.history.extend(actions)

    def maybe_forge(self, engine: "EvolutionEngine") -> Optional[Dict]:
        cand = None
        for k, c in sorted(self.fail_counts.items()):
            if c >= FORGE_FAIL_NEED:
                cand = k
                break
        if not cand:
            return None
        fam = cand.split("|")[0]
        hist = list(self.history)
        seen, cands = set(), []
        for L in (3, 2):
            counts = Counter(tuple(hist[i:i + L])
                             for i in range(len(hist) - L + 1))
            for seq, c in counts.most_common(8):
                if c >= SEQ_NEED and macro_is_reversible(list(seq)):
                    if seq not in seen and seq not in set(map(tuple, self.recipes.values())):
                        seen.add(seq)
                        cands.append((list(seq), c))
            if len(cands) >= 3:
                break
        if not cands:
            return None
        # 在高频候选中挑干跑收益最高者：无脑复读会被有用序列挤掉
        best = None
        for seq, c in cands[:4]:
            ok, delta = self._dry_run(engine.env, seq)
            if ok and (best is None or delta > best[2]):
                best = (seq, c, delta)
        if not best or best[2] < -0.02:
            return None
        seq, occ = best[0], best[1]
        name = "macro:" + fam + "_" + str(engine.next_macro_id)
        engine.next_macro_id += 1
        parent = engine.richest_of("assoc")
        if parent is None:
            return None
        child = engine.spawn_effect(parent, tool=name, recipe=seq, reason="forge")
        self.recipes[name] = seq
        self.forged += 1
        self.fail_counts[cand] = 0
        return {"name": name, "recipe": seq, "occurrences": occ,
                "parent": parent.pid, "child": child.pid}

    @staticmethod
    def _dry_run(env: ResourceEnv, seq: List[str]):
        """在克隆世界干跑；返回 (可行性, 带内贴近度变化)。"""
        clone = env.clone()
        pre = clone.prox()
        try:
            for step in seq:
                clone.apply(step)
        except Exception:
            return False, -1.0
        if abs(clone.x) > 1.0:
            return False, -1.0
        return True, clone.prox() - pre


# ======================================================================
# 指标登记簿：一切对外语句的数据锚点
# ======================================================================
class MetricsRegistry:
    def __init__(self):
        self.rows: List[Dict] = []

    def push(self, row: Dict) -> None:
        self.rows.append(row)
        if len(self.rows) > 4000:
            del self.rows[:1000]

    def series(self, key: str, last: int = 60) -> List[float]:
        return [r[key] for r in self.rows[-last:] if key in r]


# ======================================================================
# 演化引擎
# ======================================================================
class EvolutionEngine:
    KINDS = ("sensor", "assoc", "memory", "meta", "effect")

    def __init__(self, seed: int = 7, out_dir: str = "out"):
        self.rng = random.Random(seed)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.p_events = os.path.join(out_dir, "events.jsonl")
        self.p_growth = os.path.join(out_dir, "growth_log.jsonl")

        self.tick = 0
        self.births = 0
        self.deaths = 0
        self.next_pid = 1
        self.next_macro_id = 1
        self.recent_mutations: deque = deque(maxlen=400)
        self.total_replays = 0
        self.action_counts: Counter = Counter()

        self.env = ResourceEnv(self.rng)
        self.medium = FieldMedium()
        self.constraints = ConstraintStore(os.path.join(out_dir, "constraints.json"))
        self.accountability = AccountabilityWindow()
        self.forge = ToolForge()
        self.metrics = MetricsRegistry()

        self.teachings: deque = deque(maxlen=12)
        self.err_ema = 0.3
        self.err_baseline = 0.3
        self.corr_until = 0
        self.arousal = 0.0

        self.population: Dict[str, Primitive] = {}
        self._seed_population()
        self._log_event("GENESIS", tick=0, note="PRIMORDIA v4 初始种群就绪",
                        pop=len(self.population))

    # ------------------------------------------------------------------
    # 种群管理
    # ------------------------------------------------------------------
    def _new_pid(self) -> str:
        p = "p%04d" % self.next_pid
        self.next_pid += 1
        return p

    def _spawn(self, kind: str, tool: Optional[str] = None,
               recipe: Optional[List[str]] = None, energy: float = INIT_ENERGY) -> Primitive:
        g = Genome.random(self._new_pid(), kind, self.rng,
                          birth_tick=self.tick, tool=tool, recipe=recipe)
        pr = make_primitive(g, self.rng)
        pr.e = energy
        self.population[pr.pid] = pr
        return pr

    def _seed_population(self) -> None:
        plan = [("sensor", 4), ("assoc", 40), ("memory", 6), ("meta", 4)]
        for kind, n in plan:
            for _ in range(n):
                self._spawn(kind)
        base_tools = ["probe", "nudge_plus", "nudge_minus"]
        for t in base_tools:
            self._spawn("effect", tool=t)

    def kinds_count(self) -> Dict[str, int]:
        c = {k: 0 for k in self.KINDS}
        for pr in self.population.values():
            c[pr.kind] += 1
        return c

    def effects(self) -> List[Primitive]:
        return [p for p in self.population.values() if p.kind == "effect"]

    def richest_of(self, kind: str) -> Optional[Primitive]:
        pool = [p for p in self.population.values() if p.kind == kind]
        if not pool:
            return None
        return max(pool, key=lambda p: p.e)

    def spawn_effect(self, parent: Primitive, tool: str,
                     recipe: List[str], reason: str) -> Primitive:
        g = parent.genome.copy_with(self._new_pid(), self.tick)
        g.kind = "effect"
        g.tool = tool
        g.recipe = list(recipe)
        child = make_primitive(g, self.rng)
        child.e = INIT_ENERGY * 0.75
        parent.e -= INIT_ENERGY * 0.5
        self.population[child.pid] = child
        self.births += 1
        self._log_event("TOOL_FORGED", tick=self.tick, tool=tool,
                        recipe=list(recipe), parent=parent.pid, child=child.pid,
                        reason=reason)
        return child

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _log_event(self, etype: str, **kw) -> None:
        row = {"type": etype, "tick": self.tick}
        row.update(kw)
        try:
            with open(self.p_events, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + chr(10))
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 主循环一步
    # ------------------------------------------------------------------
    def step(self) -> None:
        self.tick += 1
        pop = self.population

        # -- 法则2（收入端）：环境收益按人头均分（无偏红利） --
        income = self.env.tick()
        n = len(pop)
        dividend = income / max(1, n)
        for pr in pop.values():
            pr.e = clamp(pr.e + dividend, -1.0, ENERGY_CAP)

        # -- meta 监察推送全局快照 --
        prev_income = getattr(self, "_prev_income", income)
        d_income = clamp((income - prev_income) * 2.0, -1.0, 1.0)
        self._prev_income = income
        snap = 0.6 * self.arousal + 0.4 * d_income
        for pr in pop.values():
            if pr.kind == "meta":
                pr.snapshot_drive = snap

        # -- 感受与局部动力学 --
        obs_code = self.env.encode(self.env.observe())
        fired: List[Primitive] = []
        for pr in pop.values():
            drive = pr.listen_drive(self.medium.field,
                                    obs_code if pr.kind == "sensor" else None)
            if pr.update(drive):
                pr.last_fire_tick = self.tick
                fired.append(pr)
            pr.learn(self.medium.field)
            if pr.kind == "memory":
                if pr.maybe_replay():
                    self.total_replays += 1

        # -- 无偏均值聚合：下一拍的全场 --
        self.medium.begin(len(pop))
        for pr in pop.values():
            self.medium.submit(pr.emit_contribution())
        self.medium.aggregate()

        # -- 法则1 赫伯加强：共燃缔键 --
        if fired:
            sample = fired if len(fired) <= 48 else self.rng.sample(fired, 48)
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    a, b = sample[i], sample[j]
                    a_new = b.pid not in a.bonds
                    b_new = a.pid not in b.bonds
                    if a_new and b_new and len(a.bonds) < BOND_CAP and len(b.bonds) < BOND_CAP:
                        a.add_bond(b.pid, self.tick)
                        b.add_bond(a.pid, self.tick)
                        self._log_event("BOND_FORMED", tick=self.tick,
                                        a=a.pid, b=b.pid,
                                        ka=a.kind, kb=b.kind)
                    else:
                        a.cofire(b.pid)
                        b.cofire(a.pid)

        # -- 拥挤税：超额键维护成本（法则2 的拓扑面）--
        for pr in pop.values():
            excess = len(pr.bonds) - CROWD_FREE
            if excess > 0:
                pr.e -= CROWD_TAX * excess

        # -- 法则5 侧向抑制 --
        for pr in fired:
            for other_pid in pr.bonds:
                q = pop.get(other_pid)
                if q is not None:
                    q.receive_lateral(LAT_INH)

        # -- 键衰减与拆除 --
        for pr in pop.values():
            for gone in pr.decay_bonds():
                q = pop.get(gone)
                if q is not None:
                    q.remove_bond(pr.pid)

        # -- 法则4 内在可塑性 --
        for pr in fired:
            p_mut = MUT_P_BASE * (0.5 + pr.surprise_ema)
            if self.rng.random() < p_mut:
                pr.genome.mutate(self.rng)
                self.recent_mutations.append(self.tick)

        # -- 生命周期：死亡 → 分裂 → 分化保全 → 原始汤 --
        dead = [pid for pid, pr in pop.items() if pr.e <= 0.0]
        for pid in dead:
            pr = pop.pop(pid)
            pr.alive = False
            self.deaths += 1
            for other_pid in list(pr.bonds):
                q = pop.get(other_pid)
                if q is not None:
                    q.remove_bond(pid)
        if dead:
            self._log_event("DEATHS", tick=self.tick, n=len(dead),
                            sample=dead[:5])

        born = 0
        for pr in list(pop.values()):
            if pr.e >= SPLIT_AT and len(pop) < POP_CAP:
                child = pr.mitosis_child(self._new_pid(), self.tick)
                pop[child.pid] = child
                self.births += 1
                born += 1
                if born <= 6:
                    self._log_event("MITOSIS", tick=self.tick, parent=pr.pid,
                                    child=child.pid, kind=pr.kind,
                                    gen=child.genome.gen)
        kc = self.kinds_count()
        for kind in self.KINDS:
            if kc[kind] == 0:
                src = self.richest_of("assoc") or (
                    max(pop.values(), key=lambda p: p.e) if pop else None)
                if src is not None and src.kind != kind:
                    self._convert(src, kind)
                    self._log_event("DIFFERENTIATION", tick=self.tick,
                                    pid=src.pid, to=kind,
                                    note="某分化型灭绝后由最富关联基元再分化（局部规则，非指令）")
                    kc = self.kinds_count()
        if 0 < len(pop) < MIN_POP:
            need = MIN_POP - len(pop)
            for _ in range(need):
                self._spawn("assoc", energy=INIT_ENERGY * 0.8)
            self._log_event("PRIMORDIAL_RESEED", tick=self.tick, n=need,
                            note="种群逼近消亡，注入原始汤（诚实记录的边界条件）")

        # -- 唤醒度与集体决策 --
        if pop:
            self.arousal = sum(math.tanh(max(0.0, pr.p)) for pr in pop.values()) / len(pop)
        self._maybe_quorum()

        # -- 工具锻造 --
        if self.tick % FORGE_CHECK_EVERY == 0:
            made = self.forge.maybe_forge(self)
            if made:
                pass  # 事件已在 spawn_effect 内记录

        # -- 自我修正监察 --
        self._self_correct()

        # -- 成长日志采样 --
        if self.tick % SNAPSHOT_EVERY == 0:
            self._snapshot_metrics(income)

    # ------------------------------------------------------------------
    # 集体决策：法定人数表决
    # ------------------------------------------------------------------
    def _maybe_quorum(self) -> None:
        if self.tick % QUORUM_PERIOD != 0 and self.arousal < AROUSAL_TRIGGER:
            return
        sig = self.env.context_sig()
        movers: List[tuple] = []
        for eff in self.effects():
            sup = eff.motion_support()
            if sup is None:
                continue
            fam = eff.genome.tool
            if self.constraints.vetoed(fam, sig):
                self.forge.record_fail(fam, sig)
                self._log_event("MOTION_VETOED", tick=self.tick, pid=eff.pid,
                                family=fam, sig=sig, support=round(sup, 3))
                continue
            if sup >= MOTION_MIN:
                movers.append((eff, sup))
        if not movers:
            return
        movers.sort(key=lambda t: -t[1])
        top = movers[0][1]
        ties = [m for m in movers if abs(m[1] - top) < 1e-9]
        winner, _wsup = self.rng.choice(ties)   # 并列 → 公平抽签（仍是无偏聚合）
        margin = top - (movers[1][1] if len(movers) > 1 else 0.0)
        participants = [m[0].pid for m in movers]
        # 落选但真诚的动议 = 未满足的需求：锻造炉据此孕育新工具
        for m_eff, m_sup in movers:
            if m_eff.pid != winner.pid:
                self.forge.record_fail(m_eff.genome.tool, sig)

        alliance = [winner.pid] + [q for q in winner.bonds.keys() if q in self.population]
        pre = self.env.prox()
        ctx = ActionContext(env=self.env, tick=self.tick, context_sig=sig,
                            constraints=self.constraints, confirm_token=None)
        tool = winner.genome.tool
        steps = (list(winner.genome.recipe)
                 if (tool or "").startswith("macro:") and winner.genome.recipe
                 else [tool or ""])
        results, blocked = [], None
        try:
            for step in steps:
                r = winner.execute(step, ctx)
                results.append(r)
                if not r.get("ok"):
                    if r.get("why") == "constraint_veto":
                        self.forge.record_fail(tool, sig)
                    break
        except IrreversibleBlocked as exc:
            blocked = str(exc)
            self._log_event("IRREVERSIBLE_BLOCKED", tick=self.tick,
                            pid=winner.pid, why=str(exc),
                            note="原生安全门：缺人类令牌，物理上无法执行")
        if blocked is None and results and results[-1].get("ok"):
            delta = self.env.prox() - pre
            reward = REWARD_SCALE * max(0.0, delta)
            per = reward / len(alliance) if alliance else 0.0
            for pid in alliance:
                q = self.population.get(pid)
                if q is not None:
                    q.e = clamp(q.e + per, -1.0, ENERGY_CAP)
            executed = [r["action"] for r in results]
            self.forge.record_success(executed)
            self.accountability.record(self.tick, tool, participants,
                                       winner.pid, sig, delta)
            self.action_counts[tool] += 1
            self._log_event("ACTION_EXECUTED", tick=self.tick, family=tool,
                            winner=winner.pid, margin=round(margin, 3),
                            participants=len(participants), alliance=len(alliance),
                            prox_delta=round(delta, 4),
                            reward_each=round(per, 4), steps=executed, sig=sig)
        elif blocked is None:
            self.forge.record_fail(tool, sig)
            self._log_event("ACTION_FAILED", tick=self.tick, family=tool,
                            why=results[-1].get("why") if results else "?")

    # ------------------------------------------------------------------
    # 自我修正：监察基元的惊讶度异常 ⇒ 阻尼最惊讶者
    # ------------------------------------------------------------------
    def _self_correct(self) -> None:
        metas = [p for p in self.population.values() if p.kind == "meta"]
        if not metas:
            return
        err = sum(m.surprise_ema for m in metas) / len(metas)
        self.err_ema += 0.05 * (err - self.err_ema)
        self.err_baseline += 0.0025 * (err - self.err_baseline)
        if (self.tick > 300 and self.tick > self.corr_until
                and self.err_ema > max(0.08, 1.6 * self.err_baseline)):
            hot = sorted(self.population.values(),
                         key=lambda p: -p.surprise_ema)[:3]
            for pr in hot:
                pr.genome.gain += (1.0 - pr.genome.gain) * 0.2
                pr.theta = clamp(pr.theta + 0.08, 0.05, 3.0)
            self.corr_until = self.tick + 90
            self._log_event("SELF_CORRECTION", tick=self.tick,
                            err_ema=round(self.err_ema, 4),
                            baseline=round(self.err_baseline, 4),
                            damped=[p.pid for p in hot],
                            note="监察惊讶异常 ⇒ 回拢增益、上调阈值")

    # ------------------------------------------------------------------
    # 结构转换与统计
    # ------------------------------------------------------------------
    def _convert(self, pr: Primitive, new_kind: str) -> None:
        g = pr.genome
        tool, recipe = (g.tool, g.recipe)
        if new_kind == "effect" and not tool:
            tool = self.rng.choice(["probe", "nudge_plus", "nudge_minus"])
            recipe = None
        ng = Genome(pid=g.pid, kind=new_kind, address=list(g.address),
                    gain=g.gain, theta0=g.theta0, rho_star=g.rho_star,
                    parent=g.parent, birth_tick=g.birth_tick, tool=tool,
                    recipe=recipe, mutations=g.mutations, gen=g.gen)
        np_ = make_primitive(ng, self.rng)
        np_.p, np_.theta, np_.e = pr.p, pr.theta, pr.e
        np_.bonds = pr.bonds
        np_.rate_ema, np_.comp_ema, np_.surprise_ema = (
            pr.rate_ema, pr.comp_ema, pr.surprise_ema)
        np_.fires, np_.replays = pr.fires, pr.replays
        self.population[pr.pid] = np_

    def _graph_stats(self) -> Dict:
        parent: Dict[str, str] = {}

        def find(a: str) -> str:
            while parent.get(a, a) != a:
                parent[a] = parent.get(parent[a], parent[a])
                a = parent[a]
            return a

        for pid in self.population:
            parent[pid] = pid
        bonds = 0
        deg_sum = 0
        for pid, pr in self.population.items():
            deg_sum += len(pr.bonds)
            for other in pr.bonds:
                if other in self.population and pid < other:
                    bonds += 1
                    ra, rb = find(pid), find(other)
                    if ra != rb:
                        parent[ra] = rb
        comps: Dict[str, int] = {}
        for pid in self.population:
            comps[find(pid)] = comps.get(find(pid), 0) + 1
        largest = max(comps.values()) if comps else 0
        n = max(1, len(self.population))
        return {"bonds": bonds,
                "components": len(comps),
                "largest_frac": round(largest / n, 3),
                "mean_degree": round(deg_sum / n, 2)}

    def _snapshot_metrics(self, income: float) -> None:
        kc = self.kinds_count()
        gs = self._graph_stats()
        pop_n = max(1, len(self.population))
        row = {
            "tick": self.tick,
            "pop": len(self.population),
            "by_kind": kc,
            "arousal": round(self.arousal, 4),
            "competence": round(sum(p.comp_ema for p in self.population.values()) / pop_n, 4),
            "novelty": sum(1 for t in self.recent_mutations if t > self.tick - 100),
            "pool_income": round(income, 4),
            "births": self.births, "deaths": self.deaths,
            "tools": len([p for p in self.population.values()
                          if p.kind == "effect" and p.genome.tool]),
            "replays": self.total_replays,
        }
        row.update(gs)
        self.metrics.push(row)
        try:
            with open(self.p_growth, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + chr(10))
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 对通信层暴露的事实接口
    # ------------------------------------------------------------------
    def status_facts(self) -> Dict:
        gs = self._graph_stats()
        kc = self.kinds_count()
        pop_n = max(1, len(self.population))
        fld = [round(v, 3) for v in self.medium.field]
        return {
            "tick": self.tick,
            "pop": len(self.population), "by_kind": kc,
            "bonds": gs["bonds"], "components": gs["components"],
            "largest_frac": gs["largest_frac"], "mean_degree": gs["mean_degree"],
            "arousal": round(self.arousal, 4),
            "competence": round(sum(p.comp_ema for p in self.population.values()) / pop_n, 4),
            "novelty": sum(1 for t in self.recent_mutations if t > self.tick - 100),
            "income": round(self.env.last_income, 4),
            "env": {"x": round(self.env.x, 3), "target": round(self.env.target, 3),
                    "in_band": self.env.in_band(), "stability": round(self.env.stability(), 3)},
            "births": self.births, "deaths": self.deaths,
            "field": fld, "dividend_last": round(self.env.last_income / pop_n, 5),
            "constraints_active": self.constraints.count(),
            "replays": self.total_replays,
            "actions": dict(self.action_counts),
        }

    def match_family(self, token: str) -> Optional[str]:
        known = set(ACTION_SPECS) | {p.genome.tool for p in self.population.values()
                                     if p.kind == "effect" and p.genome.tool}
        if token in known:
            return token
        for k in known:
            if k and (k in token or token in k):
                return k
        return None

    def correct(self, token: str, good: bool, reason: str = "") -> Dict:
        fam = self.match_family(token)
        acc = self.accountability.last
        touched: List[str] = []
        sig_used = ""
        if acc and (fam is None or acc["family"] == fam):
            sig_used = acc["sig"]
            for pid in acc["participants"]:
                q = self.population.get(pid)
                if q is None:
                    continue
                if good:
                    q.e = clamp(q.e + PRAISE_ENERGY, -1.0, ENERGY_CAP)
                    q.theta = clamp(q.theta - PRAISE_THETA, 0.05, 3.0)
                else:
                    q.e = clamp(q.e - FROWN_FINE, -1.0, ENERGY_CAP)
                    q.theta = clamp(q.theta + FROWN_THETA, 0.05, 3.0)
                touched.append(pid)
        if fam:
            sig = sig_used or "*"
            if good:
                self.constraints.relax(fam, sig)
            else:
                self.constraints.add(fam, sig, reason, self.tick)
        self._log_event("PRAISE" if good else "CORRECTION", tick=self.tick,
                        token=token, family=fam, touched=touched,
                        reason=reason[:60], sig=sig_used)
        return {"family": fam, "touched_n": len(touched), "sig": sig_used,
                "constraints_active": self.constraints.count()}

    def inject_teaching(self, text: str) -> Dict:
        sensors = [p for p in self.population.values() if p.kind == "sensor"]
        for pr in sensors:
            pr.inject_pulse(0.6)
        self.teachings.append({"tick": self.tick, "text": text[:40]})
        self._log_event("TEACH", tick=self.tick, text=text[:40],
                        sensors=n_sensors_safe(sensors))
        return {"sensors": len(sensors)}

    def describe_sample(self, n: int = 3) -> List[str]:
        picks = sorted(self.population.values(),
                       key=lambda p: -p.comp_ema)[:max(1, n)]
        return [p.describe(self.tick) for p in picks]

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save_snapshot(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.out_dir, "snapshot.json")
        data = {
            "version": "primordia-v4",
            "tick": self.tick, "births": self.births, "deaths": self.deaths,
            "next_pid": self.next_pid, "next_macro_id": self.next_macro_id,
            "env": self.env.state_dict(),
            "constraints": self.constraints.map,
            "forge_recipes": self.forge.recipes,
            "action_counts": dict(self.action_counts),
            "population": [],
        }
        for pr in self.population.values():
            data["population"].append({
                "genome": pr.genome.to_dict(),
                "p": round(pr.p, 4), "theta": round(pr.theta, 4),
                "e": round(pr.e, 4), "fires": pr.fires, "replays": pr.replays,
                "bonds": {k: round(b.cofire, 2) for k, b in pr.bonds.items()},
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return path

    def load_snapshot(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return False
        self.population.clear()
        self.tick = data.get("tick", 0)
        self.births = data.get("births", 0)
        self.deaths = data.get("deaths", 0)
        self.next_pid = data.get("next_pid", 1)
        self.next_macro_id = data.get("next_macro_id", 1)
        env = data.get("env", {})
        self.env.x = env.get("x", 0.0)
        self.env.target = env.get("target", 0.0)
        self.env.t = env.get("t", 0)
        self.env.hist = deque(env.get("hist", [0.02] * 32), maxlen=32)
        self.constraints.map = data.get("constraints", {})
        self.forge.recipes = data.get("forge_recipes", {})
        self.action_counts = Counter(data.get("action_counts", {}))
        for rec in data.get("population", []):
            g = Genome(**{k: v for k, v in rec["genome"].items()})
            pr = make_primitive(g, self.rng)
            pr.p, pr.theta, pr.e = rec.get("p", 0.0), rec.get("theta", g.theta0), rec.get("e", INIT_ENERGY)
            pr.fires, pr.replays = rec.get("fires", 0), rec.get("replays", 0)
            for other, cf in rec.get("bonds", {}).items():
                pr.bonds[other] = Bond(other=other, cofire=cf)
            self.population[pr.pid] = pr
        self._log_event("RESUME", tick=self.tick, pop=len(self.population),
                        note="从快照恢复")
        return True


def n_sensors_safe(sensors) -> int:
    return len(sensors)
