# -*- coding: utf-8 -*-
"""
SEED OS v2 — evolution_engine.py
演化引擎：结构在时间维度上的自我优化。

引擎不是控制器，而是自然规则的执行器。它只做三件事：
1. 每 tick 驱动基元动力学（均值场聚合 → 基元更新 → 赫布键更新）
2. 每结构周期应用五类结构事件（生长/增殖/剪枝/重组/工具涌现）
3. 记录一切（轨迹、事件、缘由）

不存在损失函数，不存在外部优化器。结构变化的唯一驱动力是五条自然规则。
"""
import json
import math
import os
import pickle
import random
import time
from collections import deque

from primitives import (
    Field, Primitive, SensoryPrimitive, NeuronPrimitive, ActuatorPrimitive,
    MirrorPrimitive, ToolPrimitive, entropy, clamp,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_VOCAB = [
    "结构", "智能", "演化", "场", "基元", "共振", "抑制", "生长", "剪枝",
    "重组", "工具", "镜像", "预测", "误差", "纠正", "观察", "沙箱", "报告",
    "structure", "intelligence", "evolution", "field", "primitive", "resonance",
    "growth", "pruning", "recombination", "tool", "mirror", "prediction",
]


def pearson(x, y):
    x, y = list(x), list(y)
    n = min(len(x), len(y))
    if n < 10:
        return 0.0
    x, y = x[-n:], y[-n:]
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return cov / math.sqrt(vx * vy)


class EvolutionEngine:
    """无中心演化引擎。所有结构决策的依据都来自基元自身的局部统计量。"""

    MAX_POP = 120          # 种群上限（普通电脑可承受）
    MAX_FIELDS = 24        # 场数量上限
    CYCLE = 40             # 结构周期（tick 数）
    T_GROW = 1.10          # 结构键生长阈值
    T_SPAWN = 1.30         # 增殖所需键强度
    T_TOOL = 0.90          # 工具涌现所需键强度
    SNAPSHOT_EVERY = 5     # 每 5 个结构周期自动快照（跨重启延续同一段生命）
    SNAPSHOT_VERSION = 2

    def __init__(self, growth_dir=None, seed=42, resume=True):
        random.seed(seed)
        self.tick = 0
        self.cycle = 0
        self.registry = {}          # pid -> Primitive
        self.fields = {}            # fid -> Field
        self.bonds = {}             # (pid_a, pid_b) -> strength（仅拓扑决策，非信号权重）
        self.events = []            # 内存事件流（同时持久化）
        self.input_queue = deque()  # 待注入符号
        self.action_proposals = []  # 行动基元本周期提案（通信层消费）
        self.pending_actions = []   # 待人类确认的不可逆动作
        self.recent_active = set()  # 近期高活性基元（纠正 valorization 用）
        self.correction = None      # 当前纠正信号 {"text":..., "remaining": N}
        self.on_cycle = None        # 周期回调（通信层注册：报告/仪表盘）
        self.dialogue_count = 0
        self.action_history = deque(maxlen=200)
        self.stim_log = deque(maxlen=240)     # (tick, pid)：外部符号注入足迹（能力归因用）
        self.active_trace = deque(maxlen=80)  # (tick, frozenset)：逐 tick 活跃快照
        self.outcome_boost_until = 0          # 好评后的赫布偏置窗口
        self.outcome_damp_until = 0           # 差评后的削弱窗口
        self.outcome_stats = {"reward": 0, "penalty": 0}

        gdir = growth_dir or os.path.join(BASE_DIR, "growth")
        self.gdir = gdir
        self.sandbox = os.path.join(gdir, "sandbox")
        os.makedirs(self.sandbox, exist_ok=True)
        self.trajectory_path = os.path.join(gdir, "trajectory.jsonl")
        self.events_path = os.path.join(gdir, "events.jsonl")
        self.growthlog_path = os.path.join(gdir, "growth_log.md")
        self.snapshot_path = os.path.join(gdir, "snapshot.pkl")
        self.metrics_history = []

        self.awakenings = 0           # 苏醒次数：跨重启延续同一段结构史
        self._pending_tool_eval = None  # 自主工具调用的效果评估挂起项

        self.vocab = list(DEFAULT_VOCAB)
        self._fid_seq = 0
        self._tool_seq = 0
        loaded = False
        if resume and os.path.exists(self.snapshot_path):
            try:
                self._load_snapshot()
                loaded = True
            except Exception as exc:
                self._log_event("awaken_failed", f"苏醒失败，退回种子新生：{exc}",
                                {"snapshot": os.path.basename(self.snapshot_path)})
        if not loaded:
            self._init_population()

    # ------------------------------------------------------------------
    # 初始结构：最小可自组织的种子拓扑
    # ------------------------------------------------------------------

    def _new_field(self, name):
        self._fid_seq += 1
        fid = f"F{self._fid_seq:03d}"
        f = Field(fid, name, self.tick)
        self.fields[fid] = f
        return f

    def _join(self, p, field):
        field.members.add(p.id)
        p.listen_fields.add(field.id)
        p.emit_fields.add(field.id)

    def _init_population(self):
        # 感知基元：分摊符号表
        vocab = list(self.vocab)
        random.shuffle(vocab)
        n_sensory = 8
        sensory = [SensoryPrimitive(vocab[i::n_sensory], self.tick) for i in range(n_sensory)]
        neurons = [NeuronPrimitive(self.tick) for _ in range(12)]
        actuators = [ActuatorPrimitive(self.tick) for _ in range(2)]
        mirror = MirrorPrimitive(self.tick)

        # 种子工具：观察便笺（人类预置，creator=human）
        seed_tool = ToolPrimitive("notepad", {"pattern": {}},
                                  self._make_notepad_op(), self.tick)
        seed_tool.creator = "human"
        seed_tool._op_kind = "notepad"

        for p in sensory + neurons + actuators + [mirror, seed_tool]:
            self.registry[p.id] = p

        # 三个种子场：感知-计算 / 计算-行动（含镜像观察）/ 纯内部计算
        f1 = self._new_field("感知计算场")
        f2 = self._new_field("计算行动场")
        f3 = self._new_field("内部计算场")
        for p in sensory:
            self._join(p, f1)
        for p in neurons[:8]:
            self._join(p, f1)
        for p in neurons[4:]:
            self._join(p, f2)
        for p in actuators:
            self._join(p, f2)
        self._join(mirror, f2)       # 镜像观察行动场
        for p in neurons[6:]:
            self._join(p, f3)
        self.mirror_id = mirror.id
        self.seed_tool_id = seed_tool.id
        self._log_event("init", "初始结构",
                        {"sensory": len(sensory), "neurons": len(neurons),
                         "actuators": len(actuators), "fields": 3})

    # ------------------------------------------------------------------
    # 输入接口
    # ------------------------------------------------------------------

    def feed_text(self, text):
        """外部文本 → 符号流。外部世界影响系统的唯一通道。"""
        tokens = [t for t in
                  text.replace("，", " ").replace("。", " ").replace(",", " ")
                      .replace("?", " ").replace("？", " ").replace(":", " ")
                      .replace("：", " ").split() if t]
        for tok in tokens:
            self.input_queue.append(tok)
        return tokens

    def handle_correction(self, text):
        """人类纠正：纳入演化方向（赫布 valorization + 感知敏感化）。"""
        self.correction = {"text": text, "remaining": 200}
        for p in self.registry.values():
            if isinstance(p, SensoryPrimitive):
                p.target_fire_rate = 0.15   # 感知敏感化
        self._log_event("correction", f"人类纠正信号：{text}",
                        {"valorization_ticks": 200, "sensory_target_rate": 0.15})
        self.feed_text(text)

    def handle_outcome(self, good, note=""):
        """能力选择入口（v2.7）：人类对一次回答的评价变成结构层的选择压力。

        好评 → 参与本轮回答的基元各记功 +1，相关通路赫布偏置 ×1.4（80 tick）；
        差评 → 各记过 -1，参与键削弱 ×0.9（120 tick）。
        功绩随周期衰减；记功者优先增殖，记过多者加速凋亡——
        「答得好」从此是结构存活的理由。"""
        t = self.tick
        win = self.CYCLE
        sens = {pid for (tk, pid) in self.stim_log if 0 <= t - tk <= win}
        act = set()
        for tk, acts in self.active_trace:
            if 0 <= t - tk <= win * 2:
                act |= acts
        targets = act | sens
        etype = "competence_reward" if good else "competence_penalty"
        if not targets:
            self._log_event(etype, "评价落地但找不到参与结构（近窗无活动）",
                            {"note": note[:40]})
            return {"targets": 0, "bonds_damped": 0}
        if good:
            self.outcome_boost_until = t + 80
            for pid in targets:
                p = self.registry.get(pid)
                if p is not None:
                    p.merit = min(10.0, p.merit + 1.0)
            self.outcome_stats["reward"] += 1
            top = max((self.registry[p].merit for p in targets
                       if p in self.registry), default=0.0)
            self._log_event(etype,
                            f"好评记功：{len(targets)} 个基元各 +1 功绩，"
                            f"参与通路赫布偏置 ×1.4（80 tick）",
                            {"targets": len(targets), "top_merit": round(top, 2),
                             "note": note[:40]})
            return {"targets": len(targets), "bonds_damped": 0}
        self.outcome_damp_until = t + 120
        tl = sorted(targets)
        hit_bonds = 0
        for a_i in range(len(tl)):
            for b_i in range(a_i + 1, len(tl)):
                key = (tl[a_i], tl[b_i])
                if key in self.bonds:
                    self.bonds[key] *= 0.9
                    hit_bonds += 1
        for pid in targets:
            p = self.registry.get(pid)
            if p is not None:
                p.merit = max(-10.0, p.merit - 1.0)
        self.outcome_stats["penalty"] += 1
        self._log_event(etype,
                        f"差评记过：{len(targets)} 个基元各 -1 功绩，"
                        f"{hit_bonds} 条参与键削弱 ×0.9（120 tick）",
                        {"targets": len(targets), "bonds_damped": hit_bonds,
                         "note": note[:40]})
        return {"targets": len(targets), "bonds_damped": hit_bonds}

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, ticks, verbose=False):
        for _ in range(ticks):
            self.step()
        return self.metrics_history[-1] if self.metrics_history else {}

    def step(self):
        self.tick += 1
        t = self.tick

        # —— 1. 外部输入注入（无输入时保持自发背景活动）——
        self._inject_input()
        self._ambient_activity()

        # —— 2. 场聚合（无偏均值）+ 抑制泄放 ——
        for f in self.fields.values():
            f.aggregate(self.registry)
            f.release_inhibition()

        # —— 3. 基元局部更新 ——
        for p in self.registry.values():
            p.step(t, self.registry, self.fields)

        # —— 4. 自然规则 1：赫布键更新（仅拓扑决策，非信号权重）——
        self._update_bonds()

        # —— 5. 纠正 valorization 衰减 ——
        if self.correction:
            self.correction["remaining"] -= 1
            if self.correction["remaining"] <= 0:
                self.correction = None
                for p in self.registry.values():
                    if isinstance(p, SensoryPrimitive):
                        p.target_fire_rate = 0.08

        # —— 6. 镜像自模型（观察→预测→误差；离散任务：预测下一最强场）——
        acts = [p.emission for p in self.registry.values()]
        gma = sum(acts) / len(acts) if acts else 0.0
        self.registry[self.mirror_id].observe_and_predict(gma, self.fields)

        # —— 7. 近期活跃集合（纠正放大目标）——
        self.recent_active = {p.id for p in self.registry.values() if p.emission > 0.45}
        self.active_trace.append((t, self.recent_active))   # 能力归因快照

        # —— 8. 行动基元提案收集 ——
        for p in self.registry.values():
            if isinstance(p, ActuatorPrimitive) and p.proposals:
                self.action_proposals.extend(p.proposals)
                for pr in p.proposals:
                    self.action_history.append(pr["action"])

        # —— 9. 结构周期：五类结构事件 + 指标 + 记录 ——
        if self.tick % self.CYCLE == 0:
            self._structural_cycle()
            self._compute_metrics()
            self._persist_cycle()
            if self.on_cycle:
                self.on_cycle(self)

    def _inject_input(self):
        while self.input_queue:
            tok = self.input_queue.popleft()
            for p in self.registry.values():
                if isinstance(p, SensoryPrimitive) and tok in p.symbols:
                    p.stimulate(0.9)
                    self.stim_log.append((self.tick, p.id))   # 能力归因足迹

    def _ambient_activity(self):
        """自发背景活动：无外部指令时结构仍在演化（验收标准 2）。"""
        if random.random() < 0.35:
            sensors = [p for p in self.registry.values() if isinstance(p, SensoryPrimitive)]
            if sensors:
                random.choice(sensors).stimulate(0.55)
        if random.random() < 0.12:
            neurons = [p for p in self.registry.values() if isinstance(p, NeuronPrimitive)]
            if neurons:
                random.choice(neurons).potential = clamp(
                    random.choice(neurons).potential + 0.30, 0.0, 2.0)

    def _update_bonds(self):
        """赫布加强：同场共激 → 结构键累积；全局缓慢衰减（新陈代谢）。"""
        eta, decay = 0.08, 0.004
        valor = 2.2 if self.correction else 1.0
        outcome_mult = 1.4 if self.tick < self.outcome_boost_until else (
            0.55 if self.tick < self.outcome_damp_until else 1.0)
        for f in self.fields.values():
            members = sorted(f.members)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    key = (a, b)
                    pa, pb = self.registry.get(a), self.registry.get(b)
                    if pa is None or pb is None:
                        continue
                    delta = eta * pa.emission * pb.emission
                    if valor > 1.0 and a in self.recent_active and b in self.recent_active:
                        delta *= valor   # 纠正时刻的活跃通路获得结构性偏置
                    if outcome_mult != 1.0 and a in self.recent_active and b in self.recent_active:
                        delta *= outcome_mult   # 好评记功 / 差评削弱的通路偏置
                    v = self.bonds.get(key, 0.0) + delta - decay
                    self.bonds[key] = v if v > 0 else 0.0

    # ------------------------------------------------------------------
    # 结构周期：生长 / 增殖 / 剪枝 / 重组 / 工具涌现 / 自我修正
    # ------------------------------------------------------------------

    def _structural_cycle(self):
        self.cycle += 1
        for p in self.registry.values():
            p.merit *= 0.97   # 功过随时间淡去（新陈代谢，不搞终身制）
            if isinstance(p, NeuronPrimitive) and p.spawn_cooldown > 0:
                p.spawn_cooldown -= 1   # 增殖不应期随周期流逝
        self._recombine_fields()
        self._grow_fields()
        self._proliferate()
        self._prune()
        self._tool_emergence()
        self._self_correct()
        self._maybe_use_tools()
        self._evaluate_tool_use()
        if self.cycle % self.SNAPSHOT_EVERY == 0:
            self.save_snapshot()   # 每 5 个周期自动快照，重启即苏醒

    def _grow_fields(self):
        """生长：结构键持续超阈的基元对 → 从大场分化出独立新场。"""
        if len(self.fields) >= self.MAX_FIELDS:
            return
        grew = []
        for (a, b), v in sorted(self.bonds.items(), key=lambda kv: -kv[1]):
            if v < self.T_GROW or len(grew) >= 2:
                continue
            if a not in self.registry or b not in self.registry:
                continue
            shared = self.registry[a].emit_fields & self.registry[b].emit_fields
            # 已共享专属小场（≤4 成员）的对不再重复分化
            if any(fid in self.fields and len(self.fields[fid].members) <= 4
                   for fid in shared):
                continue
            # 键强但未被同一小场覆盖 → 从大场分化新场
            if not shared or any(fid in self.fields and len(self.fields[fid].members) > 4
                                 for fid in shared):
                f = self._new_field(f"分化场C{self.cycle}")
                self._join(self.registry[a], f)
                self._join(self.registry[b], f)
                self.bonds[(a, b)] = v * 0.5   # 消耗键强度（结构投资）
                grew.append((a, b, v))
                self._log_event("growth", f"{a} 与 {b} 分化出新场 {f.id}",
                                {"bond": round(v, 3), "field": f.id})
        return grew

    def _proliferate(self):
        """增殖：高能量 + 多超阈键的计算基元分裂出参数微扰的子代。
        v2.7 能力选择：合格者中功绩最高者优先繁殖——答得好成为传宗接代的理由。"""
        pop = len(self.registry)
        if pop >= self.MAX_POP:
            return
        best = None
        for pid, p in list(self.registry.items()):
            if (not isinstance(p, NeuronPrimitive)
                    or p.energy < 0.32 or p.spawn_cooldown > 0):
                continue   # 能量不足或在增殖不应期：机会让给别的成员
            strong = sum(1 for (a, b), v in self.bonds.items()
                         if v > self.T_SPAWN and pid in (a, b))
            if strong < 2:
                continue
            key = (p.merit, strong, p.energy)
            if best is None or key > best[0]:
                best = (key, pid, p, strong)
        if best is None:
            return
        _, pid, p, strong = best
        child = NeuronPrimitive(self.tick)
        child.spawn_cooldown = 4   # 子代成熟期：先长身体再谈繁殖
        child.merit = p.merit * 0.5   # 血统声望减半继承，功过不搞世袭
        child.theta = clamp(p.theta + random.uniform(-0.06, 0.06), 0.15, 0.95)
        child.gain = clamp(p.gain + random.uniform(-0.15, 0.15), 0.5, 3.0)
        child.listen_fields = set(p.listen_fields)
        child.emit_fields = set(p.emit_fields)
        for fid in child.emit_fields:
            if fid in self.fields:
                self.fields[fid].members.add(child.id)
        self.registry[child.id] = child
        self.bonds[(min(pid, child.id), max(pid, child.id))] = 0.6
        # 增殖消耗父代结构键（结构投资）+ 父代进入不应期：不做克隆流水线
        for key in [k for k in self.bonds if pid in k]:
            self.bonds[key] *= 0.5
        p.spawn_cooldown = 8
        self._log_event("proliferation", f"{pid} 增殖出子代 {child.id}",
                        {"parent_energy": round(p.energy, 3),
                         "parent_merit": round(p.merit, 2),
                         "strong_bonds": strong, "population": len(self.registry),
                         "parent_cooldown_cycles": 8})

    def _prune(self):
        """剪枝：长窗口近零活性的计算基元移除；空场/低流量场解散。"""
        # —— 基元凋亡（仅计算基元可被移除；感知/行动/镜像/工具为种子基础设施）——
        for pid, p in list(self.registry.items()):
            if not isinstance(p, NeuronPrimitive):
                continue
            dormant = getattr(p, "dormant_cycles", 0)
            if p.energy < 0.015:
                p.dormant_cycles = dormant + 1
                # v2.7：记过多的个体休眠 1 个周期即凋亡（能力选择的选择压力）
                need = 1 if getattr(p, "merit", 0.0) <= -3.0 else 3
                if p.dormant_cycles >= need:
                    for fid in list(p.emit_fields | p.listen_fields):
                        f = self.fields.get(fid)
                        if f:
                            f.members.discard(pid)
                    self.registry.pop(pid, None)
                    for key in [k for k in self.bonds if pid in k]:
                        del self.bonds[key]
                    detail = (f"{pid} 长期休眠且记过，结构移除" if need == 1
                              else f"{pid} 长期休眠，结构移除")
                    self._log_event("pruning", detail,
                                    {"energy": round(p.energy, 4),
                                     "merit": round(getattr(p, "merit", 0.0), 2),
                                     "dormant_cycles": p.dormant_cycles})
            else:
                p.dormant_cycles = 0
        # —— 场解散 ——
        for fid, f in list(self.fields.items()):
            low_traffic = f.traffic < 0.5 * max(1, self.cycle - f.birth_tick // self.CYCLE + 1)
            if len(f.members) < 2 or (f.traffic / max(1, self.tick - f.birth_tick) < 0.003
                                      and self.tick - f.birth_tick > self.CYCLE * 3):
                self._dissolve_field(fid, "成员不足" if len(f.members) < 2 else "流量过低")

    def _dissolve_field(self, fid, reason):
        f = self.fields.pop(fid, None)
        if f is None:
            return
        members = list(f.members)
        for pid in members:
            p = self.registry.get(pid)
            if p:
                p.listen_fields.discard(fid)
                p.emit_fields.discard(fid)
                # 失连基元重新接入最繁忙的场（保持连通）
                if not p.listen_fields:
                    rest = sorted(self.fields.values(), key=lambda x: -x.traffic)
                    if rest:
                        self._join(p, rest[0])
        self._log_event("dissolve", f"场 {fid} 解散（{reason}）",
                        {"members": members, "traffic": round(f.traffic, 2)})

    def _recombine_fields(self):
        """重组：高相关场合并；场内活性双簇分布 → 裂变。"""
        fids = list(self.fields.keys())
        # —— 合并 ——
        merged = False
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                fa, fb = self.fields.get(fids[i]), self.fields.get(fids[j])
                if not fa or not fb:
                    continue
                c = pearson(fa.history, fb.history)
                if c > 0.92 and len(fa.members) + len(fb.members) <= 30:
                    reason = {"corr": round(c, 4), "field_a": fa.id, "field_b": fb.id}
                    for pid in list(fb.members):
                        p = self.registry.get(pid)
                        if p:
                            self._join(p, fa)
                            p.listen_fields.discard(fb.id)
                            p.emit_fields.discard(fb.id)
                    self.fields.pop(fb.id, None)
                    self._log_event("merge", f"场 {fa.id} 与 {fb.id} 高相关合并",
                                    reason)
                    merged = True
                    break
            if merged:
                break
        # —— 裂变：场内成员能量呈双簇（活跃簇 vs 沉默簇）——
        if len(self.fields) < self.MAX_FIELDS:
            for fid, f in list(self.fields.items()):
                if len(f.members) < 6:
                    continue
                energies = [(pid, self.registry[pid].energy)
                            for pid in f.members if pid in self.registry]
                hi = [pid for pid, e in energies if e > 0.30]
                lo = [pid for pid, e in energies if e < 0.03]
                if len(hi) >= 2 and len(lo) >= 2:
                    nf = self._new_field(f"裂变场C{self.cycle}")
                    for pid in hi:
                        p = self.registry[pid]
                        self._join(p, nf)
                        f.members.discard(pid)
                        p.listen_fields.discard(fid)
                        p.emit_fields.discard(fid)
                    self._log_event("fission", f"场 {fid} 活性双簇裂变出 {nf.id}",
                                    {"hi_cluster": hi, "lo_cluster": lo,
                                     "field": fid, "new_field": nf.id})
                    break

    def _tool_emergence(self):
        """工具涌现：含行动基元的稳定共激簇 → 固化为工具基元（系统创建工具）。"""
        tools = [p for p in self.registry.values() if isinstance(p, ToolPrimitive)]
        if len(tools) >= 10:
            return
        for fid, f in self.fields.items():
            members = [pid for pid in f.members if pid in self.registry]
            has_act = any(isinstance(self.registry[m], ActuatorPrimitive) for m in members)
            if not has_act or len(members) < 4:
                continue
            core_pairs = 0
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if self.bonds.get((members[i], members[j]), 0) > self.T_TOOL:
                        core_pairs += 1
            if core_pairs < 3:
                continue
            # —— 涌现前先验重：与既有工具配方高度重合的簇不是新创造 ——
            new_set = set(members)
            for t0 in self.registry.values():
                if isinstance(t0, ToolPrimitive) and t0.creator == "system":
                    old_set = set(t0.recipe.get("pattern", {}))
                    union = new_set | old_set
                    jac = len(new_set & old_set) / len(union) if union else 0.0
                    if jac > 0.75:
                        self._log_event("tool_skip_duplicate",
                                        f"场 {fid} 候选簇与工具 {t0.name} 配方重合度 "
                                        f"{jac:.2f}，不再固化重复工具",
                                        {"field": fid, "existing": t0.name,
                                         "max_jaccard": round(jac, 3)})
                        return
            # —— 涌现：把当前簇拓扑固化为工具配方 ——
            self._tool_seq += 1
            name = f"emergent_tool_{self._tool_seq:02d}"
            pattern = {pid: round(self.registry[pid].energy, 3) for pid in members}
            tool = ToolPrimitive(name, {"pattern": pattern, "source_field": fid},
                                 self._make_emergent_op(name, pattern), self.tick)
            self.registry[tool.id] = tool
            self._log_event("tool_emergence",
                            f"场 {fid} 稳定共激簇固化为工具 {name}",
                            {"core_pairs": core_pairs, "members": members,
                             "tool": tool.id, "invoked_via": "通信层『调用』"})
            break

    def _self_correct(self):
        """自我修正：镜像离散预测准确率过低 → 阻尼最异常基元（增益回拢 + 阈值上调）。"""
        mirror = self.registry.get(self.mirror_id)
        if not mirror or len(mirror._field_hits) < 50:
            return   # 样本不足，不做判断
        if mirror.accuracy >= 0.20:
            return
        emissions = [p.emission for p in self.registry.values()
                     if isinstance(p, NeuronPrimitive)]
        if not emissions:
            return
        med = sorted(emissions)[len(emissions) // 2]
        worst, wd = None, -1.0
        for p in self.registry.values():
            if not isinstance(p, NeuronPrimitive):
                continue   # 只修正动力学基元，不碰感知/行动/镜像/工具基础设施
            d = abs(p.emission - med)
            if d > wd:
                worst, wd = p, d
        if worst is None:
            return
        worst.gain = clamp(worst.gain * 0.8 + 0.2 * 1.0, 0.5, 3.0)   # 向 1.0 回拢
        worst.theta = clamp(worst.theta + 0.05, 0.15, 0.95)
        self._log_event("self_correction",
                        f"镜像离散预测准确率 {mirror.accuracy:.3f} 过低，"
                        f"阻尼异常基元 {worst.id}",
                        {"prediction_accuracy": round(mirror.accuracy, 4),
                         "prediction_error": round(mirror.prediction_error, 4),
                         "anomaly": worst.id, "deviation": round(wd, 4)})

    # ------------------------------------------------------------------
    # 指标与记录
    # ------------------------------------------------------------------

    def _compute_metrics(self):
        prim = list(self.registry.values())
        # 结构复杂度：场成员分布熵 + 场数因子
        sizes = [len(f.members) for f in self.fields.values()]
        total = sum(sizes) or 1
        if sizes:
            se = -sum((s / total) * math.log(s / total + 1e-12) for s in sizes)
            se_n = se / math.log(len(sizes) + 1e-12) if len(sizes) > 1 else 0.0
        else:
            se_n = 0.0
        structure_entropy = clamp(0.7 * se_n + 0.3 * min(1.0, len(sizes) / 12))
        # 表达多样性
        activity_entropy = entropy([p.emission for p in prim])
        # 自我模型准确率
        mirror = self.registry[self.mirror_id]
        prediction_acc = clamp(mirror.accuracy)
        # 行为新颖度：相邻动作二元组的去重率（序列多样性，不再恒等于 1.0）
        window = list(self.action_history)[-self.CYCLE:]
        if len(window) >= 2:
            pairs = list(zip(window, window[1:]))
            novelty = clamp(len(set(pairs)) / len(pairs))
        else:
            novelty = 0.0
        intelligence = clamp(0.30 * structure_entropy + 0.25 * activity_entropy +
                             0.25 * prediction_acc + 0.20 * novelty)
        self.metrics = {
            "tick": self.tick, "cycle": self.cycle,
            "population": len(prim), "fields": len(self.fields),
            "structure_entropy": round(structure_entropy, 4),
            "activity_entropy": round(activity_entropy, 4),
            "prediction_accuracy": round(prediction_acc, 4),
            "behavior_novelty": round(novelty, 4),
            "intelligence": round(intelligence, 4),
            "mirror_error": round(mirror.prediction_error, 4),
            "bonds_over_threshold": sum(1 for v in self.bonds.values() if v > self.T_GROW),
            "dialogue_count": self.dialogue_count,
        }
        self.metrics_history.append(self.metrics)

    def _log_event(self, etype, detail, reason=None):
        ev = {"tick": self.tick, "type": etype, "detail": detail, "reason": reason or {}}
        self.events.append(ev)
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def _persist_cycle(self):
        with open(self.trajectory_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(self.metrics, ensure_ascii=False) + "\n")
        m = self.metrics
        recent = self.events[-8:]
        lines = [
            f"\n## 周期 {m['cycle']}（tick {m['tick']}）",
            f"- 种群 {m['population']} | 场 {m['fields']} | 智能指数 "
            f"**{m['intelligence']}**（结构 {m['structure_entropy']} · "
            f"活性熵 {m['activity_entropy']} · 预测 {m['prediction_accuracy']} · "
            f"新颖度 {m['behavior_novelty']}）",
        ]
        for ev in recent:
            lines.append(f"- [{ev['type']}] {ev['detail']}（{json.dumps(ev['reason'], ensure_ascii=False)}）")
        with open(self.growthlog_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        # 仪表盘由通信层通过 on_cycle 回调生成

    # ------------------------------------------------------------------
    # 工具算子工厂（闭包不可腌制，苏醒时按 _op_kind 重建）
    # ------------------------------------------------------------------

    def _make_notepad_op(self):
        engine = self

        def _notepad_op(payload):
            note = "便笺 #%d | tick=%d | %s" % (
                random.randrange(9999), engine.tick,
                payload.get("note", "（无内容）"))
            path = os.path.join(engine.sandbox, "notepad.txt")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(note + "\n")
            return {"written": note, "file": path}

        return _notepad_op

    def _make_emergent_op(self, name, pattern):
        engine = self

        def _op(payload, _pattern=pattern, _name=name):
            path = os.path.join(engine.sandbox, f"{_name}_artifact.txt")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "tick": engine.tick, "pattern": _pattern,
                    "payload": payload}, ensure_ascii=False) + "\n")
            return {"artifact": path, "pattern_size": len(_pattern)}

        return _op

    # ------------------------------------------------------------------
    # v2.4 自主工具使用：镜像误差超带 → 重放最匹配的稳定结构模式
    # ------------------------------------------------------------------

    def _maybe_use_tools(self):
        """系统自己调用自己的工具。触发器是内部信号（镜像预测误差），
        选择依据是配方与当前高活性成员的拓扑重叠度——全程可逆、沙箱内。"""
        mirror = self.registry.get(self.mirror_id)
        if not mirror or mirror.prediction_error <= 0.12:
            return   # 预测在容差带内：不需要重放任何模式
        hot = {pid for pid, p in self.registry.items()
               if not isinstance(p, ToolPrimitive) and p.energy > 0.25}
        best, best_ov = None, 0.0
        for p in self.registry.values():
            if not isinstance(p, ToolPrimitive) or p.creator != "system":
                continue
            last = getattr(p, "_last_self_invoke_cycle", -999)
            if self.cycle - last < 10:
                continue   # 自调用冷却：重放是修正手段，不是习惯动作
            pat = {pid for pid in p.recipe.get("pattern", {}) if pid in self.registry}
            if not pat:
                continue
            ov = len(pat & hot) / len(pat)
            if ov > best_ov:
                best, best_ov = p, ov
        if best is None or best_ov < 0.55:
            return   # 没有足够匹配的工具：不硬用
        err_before = round(mirror.prediction_error, 4)
        result = best.invoke({"by": "self_initiated", "trigger": "mirror_error"})
        best._last_self_invoke_cycle = self.cycle
        self._pending_tool_eval = {"tool": best.id, "error_before": err_before,
                                   "cycle": self.cycle}
        self._log_event("self_tool_invocation",
                        f"镜像预测误差 {err_before} 超容差带，"
                        f"自主调用工具 {best.name} 重放稳定结构模式",
                        {"tool": best.id, "overlap": round(best_ov, 3),
                         "mirror_error": err_before, "result": str(result)[:100]})

    def _evaluate_tool_use(self):
        """工具效果评估：对比自调用前后的镜像误差；改善 → 配方成员结构键加强
        （内在强化，赫布式；键只参与拓扑决策，仍非信号权重）。"""
        pend = self._pending_tool_eval
        if not pend or self.cycle <= pend["cycle"]:
            return
        self._pending_tool_eval = None
        mirror = self.registry.get(self.mirror_id)
        tool = self.registry.get(pend["tool"])
        if not mirror or not isinstance(tool, ToolPrimitive):
            return
        err_after = round(mirror.prediction_error, 4)
        improved = err_after < pend["error_before"]
        reason = {"tool": tool.id, "error_before": pend["error_before"],
                  "error_after": err_after, "improved": improved}
        detail = (f"工具 {tool.name} 自调用后镜像误差 "
                  f"{pend['error_before']}→{err_after}，"
                  + ("配方成员结构键获得加强" if improved else "未见改善，不做加强"))
        if improved:
            pat = [pid for pid in tool.recipe.get("pattern", {}) if pid in self.registry]
            strengthened = 0
            for i in range(len(pat)):
                for j in range(i + 1, len(pat)):
                    key = (min(pat[i], pat[j]), max(pat[i], pat[j]))
                    self.bonds[key] = self.bonds.get(key, 0.0) + 0.25
                    strengthened += 1
            reason["strengthened_pairs"] = strengthened
        self._log_event("self_tool_evaluation", detail, reason)

    # ------------------------------------------------------------------
    # v2.4 拒绝反馈落地：「负向纠正」从台词变成结构事实
    # ------------------------------------------------------------------

    def apply_rejection_feedback(self, source_id):
        """人类拒绝不可逆动作 → 来源行动基元阈值上浮 + 进入长提案冷却。"""
        p = self.registry.get(source_id)
        if p is None:
            return None
        p.theta = clamp(p.theta + 0.03, 0.15, 0.95)
        cooldown = getattr(p, "ir_cooldown", None)
        if cooldown is not None:
            p.ir_cooldown = max(p.ir_cooldown, 600)
        self._log_event("rejection_feedback",
                        f"人类否决信号落地：{source_id} 阈值上浮并进入提案长冷却",
                        {"source": source_id, "theta_after": round(p.theta, 4),
                         "cooldown_ticks": getattr(p, "ir_cooldown", 0)})
        return source_id

    # ------------------------------------------------------------------
    # v2.4 结构快照：跨重启延续同一段生命（智能即历史，历史即连续）
    # ------------------------------------------------------------------

    def save_snapshot(self):
        """把当前全部结构状态写入 snapshot.pkl（原子替换）。"""
        try:
            state = {
                "version": self.SNAPSHOT_VERSION,
                "tick": self.tick, "cycle": self.cycle,
                "fid_seq": self._fid_seq, "tool_seq": self._tool_seq,
                "declared_count": getattr(self, "declared_count", 0),
                "auto_rejected": getattr(self, "auto_rejected", 0),
                "dialogue_count": self.dialogue_count,
                "awakenings": self.awakenings,
                "next_primitive_id": Primitive._NEXT_ID[0],
                "mirror_id": self.mirror_id,
                "registry": self.registry,
                "fields": self.fields,
                "bonds": {f"{k[0]}|{k[1]}": v for k, v in self.bonds.items()},
                "pending_actions": self.pending_actions,
                "action_history": list(self.action_history)[-200:],
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            tmp = self.snapshot_path + ".tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.snapshot_path)
            return True
        except Exception as exc:
            self._log_event("snapshot_failed", f"结构快照失败：{exc}", {})
            return False

    def _load_snapshot(self):
        with open(self.snapshot_path, "rb") as fh:
            state = pickle.load(fh)
        if state.get("version") != self.SNAPSHOT_VERSION:
            raise ValueError(f"快照版本不兼容：{state.get('version')}")
        self.registry = state["registry"]
        self.fields = state["fields"]
        self.bonds = {}
        for k, v in state["bonds"].items():
            a, b = k.split("|")
            self.bonds[(a, b)] = v
        # 重建工具算子（闭包随进程消亡，结构记忆不随之消亡）
        for p in self.registry.values():
            if isinstance(p, ToolPrimitive):
                if getattr(p, "_op_kind", "emergent") == "notepad":
                    p.set_op(self._make_notepad_op())
                else:
                    p.set_op(self._make_emergent_op(p.name, p.recipe.get("pattern", {})))
            if not hasattr(p, "merit"):     # v2.7 之前的快照个体补功绩字段
                p.merit = 0.0
        self.tick = state["tick"]
        self.cycle = state["cycle"]
        self._fid_seq = state["fid_seq"]
        self._tool_seq = state["tool_seq"]
        self.declared_count = state.get("declared_count", 0)
        self.auto_rejected = state.get("auto_rejected", 0)
        self.dialogue_count = state.get("dialogue_count", 0)
        self.awakenings = state.get("awakenings", 0) + 1
        Primitive._NEXT_ID[0] = state.get("next_primitive_id", 0)
        self.mirror_id = state["mirror_id"]
        self.pending_actions = state.get("pending_actions", [])
        self.action_history = deque(state.get("action_history", []), maxlen=200)
        # 轨迹接续：仪表盘曲线跨重启连续
        if os.path.exists(self.trajectory_path):
            with open(self.trajectory_path, "r", encoding="utf-8") as fh:
                tail = fh.readlines()[-120:]
            self.metrics_history = []
            for line in tail:
                try:
                    self.metrics_history.append(json.loads(line))
                except Exception:
                    pass
        kinds = {}
        for p in self.registry.values():
            kinds[p.KIND] = kinds.get(p.KIND, 0) + 1
        self._log_event("awaken",
                        f"苏醒：延续既有结构（tick {self.tick} / 周期 {self.cycle}）",
                        {"population": len(self.registry),
                         "fields": len(self.fields), "kinds": kinds,
                         "awakenings": self.awakenings,
                         "saved_at": state.get("saved_at")})

    # ------------------------------------------------------------------
    # 查询接口（通信层使用）
    # ------------------------------------------------------------------

    def why(self, query):
        """为什么：检索事件日志，返回最近相关结构事件及量化缘由。"""
        hits = [e for e in self.events
                if query in e["detail"] or query in json.dumps(e["reason"], ensure_ascii=False)]
        return hits[-3:][::-1]

    def invoke_tool(self, name, payload=None):
        for p in self.registry.values():
            if isinstance(p, ToolPrimitive) and p.name == name:
                result = p.invoke(payload)
                self._log_event("tool_invocation",
                                f"工具 {name} 被调用",
                                {"invocations": p.invocations, "result": str(result)[:120]})
                return result
        return None

    def list_tools(self):
        return [p.describe() for p in self.registry.values() if isinstance(p, ToolPrimitive)]

    def describe_state(self):
        prim = list(self.registry.values())
        kinds = {}
        for p in prim:
            kinds[p.KIND] = kinds.get(p.KIND, 0) + 1
        return {
            "tick": self.tick, "cycle": self.cycle,
            "population": len(prim), "kinds": kinds,
            "fields": {fid: f.describe() for fid, f in self.fields.items()},
            "metrics": getattr(self, "metrics", {}),
            "mirror": self.registry[self.mirror_id].describe(),
            "pending_actions": self.pending_actions,
            "correction_active": bool(self.correction),
            "outcome_stats": self.outcome_stats,
            "mean_merit": round(sum(getattr(p, "merit", 0.0)
                                    for p in prim) / max(1, len(prim)), 3),
            "recent_events": self.events[-10:],
        }
