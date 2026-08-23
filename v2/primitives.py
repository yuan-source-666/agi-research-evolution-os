# -*- coding: utf-8 -*-
"""
SEED OS v2 — primitives.py
基元定义与实现：结构即智能的最小单元。

所有基元是自治动力学实体：自我描述、局部状态更新、对外交互。
基元之间只通过均值场（Field）做无偏聚合交互——没有权重、路由、注意力。
"""
import math
import random
from collections import deque


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def entropy(values, bins=8):
    """归一化香农熵：values ∈ [0,1] 的分布熵，用于活性/结构多样性度量。"""
    if not values:
        return 0.0
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int(clamp(v) * bins))] += 1
    n = len(values)
    e = 0.0
    for c in counts:
        if c:
            p = c / n
            e -= p * math.log(p)
    return e / math.log(bins)


class Field:
    """均值场：成员发射信号的无偏算术均值。这是全系统唯一的信号通道。

    没有权重、没有路由、没有注意力：一个基元要么在场里，要么不在。
    结构的全部信息保存在拓扑（成员关系）中。
    """

    def __init__(self, fid, name, birth_tick=0):
        self.id = fid
        self.name = name
        self.birth_tick = birth_tick
        self.members = set()          # primitive ids（发射且订阅一体）
        self.signal = 0.0             # 本 tick 均值信号
        self.inhibition = 0.0         # 侧向抑制电流（由发放成员注入）
        self.inhibition_share = 0.0   # 本 tick 均分给每个成员的抑制量（引擎每 tick 泄放一次）
        self.history = deque(maxlen=200)   # 信号历史（重组依据）
        self.traffic = 0.0            # 累计流量（低流量场会被剪枝）

    def aggregate(self, registry):
        """无偏均值聚合：每个成员平等参与。"""
        if not self.members:
            self.signal = 0.0
            return 0.0
        s = 0.0
        for pid in self.members:
            p = registry.get(pid)
            if p is not None:
                s += p.emission
        self.signal = s / len(self.members)
        self.history.append(self.signal)
        self.traffic += self.signal
        return self.signal

    def inject_inhibition(self, amount):
        """发放成员向场注入抑制电流，均分给其他成员（侧向抑制，无偏）。"""
        self.inhibition += amount

    def release_inhibition(self):
        """引擎每 tick 调用一次：抑制电流均分给成员（侧向抑制，无偏），余量半衰。"""
        if self.members:
            self.inhibition_share = self.inhibition / len(self.members)
        else:
            self.inhibition_share = 0.0
        self.inhibition *= 0.5

    def describe(self):
        return {
            "id": self.id, "name": self.name,
            "members": sorted(self.members),
            "signal": round(self.signal, 4),
            "mean_signal": round(sum(self.history) / max(1, len(self.history)), 4),
            "traffic": round(self.traffic, 2),
            "age": self.birth_tick,
        }


class Primitive:
    """基元基类：自治动力学实体。子类只需实现 prepare()/react() 钩子。"""

    KIND = "primitive"
    _NEXT_ID = [0]

    def __init__(self, birth_tick=0):
        cls = type(self)
        cls._NEXT_ID[0] += 1
        self.id = f"{self.KIND[0].upper()}{cls._NEXT_ID[0]:03d}"
        self.birth_tick = birth_tick

        # —— 共有动力学状态 ——
        self.emission = 0.0        # 当前发射信号（分级）
        self.potential = 0.0       # 膜电位（积分-泄漏）
        self.theta = 0.55          # 发放阈值（自适应）
        self.gain = 1.0            # 传递增益（内在可塑性）
        self.energy = 0.0          # 滚动平均活性（增殖/剪枝依据）
        self.refractory = 0        # 不应期
        self.fire_count = 0
        self.firing = False

        # —— 均值互作拓扑（成员关系，非权重） ——
        self.listen_fields = set()
        self.emit_fields = set()

        # —— 自适应参数 ——
        self.target_fire_rate = 0.08   # 目标发放率
        self.merit = 0.0               # 功绩：参与好答案 +1 / 差答案 -1，随周期衰减
        self._recent_fire = deque(maxlen=100)
        self._recent_out = deque(maxlen=100)

    # ---------- 生命周期 ----------

    def step(self, tick, registry, fields):
        """局部状态更新。输入 = 订阅场信号的无偏均值 − 侧向抑制。"""
        if self.refractory > 0:
            self.refractory -= 1

        listen = [fields[fid].signal for fid in self.listen_fields if fid in fields]
        inh = (sum(fields[fid].inhibition_share for fid in self.listen_fields
                   if fid in fields) / len(listen)) if listen else 0.0
        base_input = (sum(listen) / len(listen)) if listen else 0.0
        raw_input = clamp(base_input - inh)

        # 积分-泄漏膜电位（上限防饱和失控）
        self.potential = min(0.72 * self.potential + raw_input, 2.5)
        # 分级发射（无加权，无非线性路由）
        self.emission = clamp(math.tanh(self.gain * self.potential) if self.potential > 0 else 0.0)

        self.firing = False
        if self.refractory == 0 and self.potential > self.theta:
            self.firing = True
            self.fire_count += 1
            self.refractory = 3
            # 发放 → 向所属场注入与发放强度成比例的侧向抑制电流（泄放时均分）
            for fid in self.emit_fields & self.listen_fields:
                if fid in fields:
                    fields[fid].inject_inhibition(0.8 * self.emission)
            # 发放 → 竞争抑制：直接抬高同场其他成员阈值（瞬时，自然衰减见下）
            self._competitive_inhibition(registry, fields)

        # —— 自然规则 3：自适应阈值（homeostasis）——
        self._recent_fire.append(1.0 if self.firing else 0.0)
        fr = sum(self._recent_fire) / len(self._recent_fire)
        self.theta = clamp(self.theta + 0.02 * (fr - self.target_fire_rate), 0.15, 0.95)

        # —— 自然规则 4：内在可塑性（输出均值保持在活性带）——
        self._recent_out.append(self.emission)
        mo = sum(self._recent_out) / len(self._recent_out)
        if mo < 0.30:
            self.gain = clamp(self.gain * 1.03, 0.5, 3.0)
        elif mo > 0.70:
            self.gain = clamp(self.gain * 0.97, 0.5, 3.0)

        # 滚动能量
        self.energy = 0.99 * self.energy + 0.01 * self.emission

    def _competitive_inhibition(self, registry, fields):
        """自然规则 2：竞争抑制——发放者瞬时抬高同场成员阈值。"""
        for fid in self.emit_fields:
            f = fields.get(fid)
            if not f:
                continue
            for pid in f.members:
                if pid == self.id:
                    continue
                q = registry.get(pid)
                if q is not None:
                    q.theta = clamp(q.theta + 0.015, 0.15, 0.95)

    # ---------- 自我描述 ----------

    def describe(self):
        fr = sum(self._recent_fire) / max(1, len(self._recent_fire))
        return {
            "id": self.id, "kind": self.KIND,
            "emission": round(self.emission, 4),
            "potential": round(self.potential, 4),
            "theta": round(self.theta, 4),
            "gain": round(self.gain, 4),
            "energy": round(self.energy, 4),
            "fire_rate": round(fr, 4),
            "age": self.birth_tick,
            "listen": sorted(self.listen_fields),
            "emit": sorted(self.emit_fields),
        }


class SensoryPrimitive(Primitive):
    """感知基元：系统输入的单向阀门。持有符号表，外部符号到达即激活。"""
    KIND = "sensory"

    def __init__(self, symbols, birth_tick=0):
        super().__init__(birth_tick)
        self.symbols = set(symbols)

    def stimulate(self, strength=1.0):
        """外部符号注入：直接抬高电位。这是外部世界影响系统的唯一通道。"""
        self.potential = clamp(self.potential + strength, 0.0, 2.0)

    def describe(self):
        d = super().describe()
        d["symbols"] = sorted(self.symbols)
        return d


class NeuronPrimitive(Primitive):
    """计算基元：纯动力学实体，结构生长的主要材料。"""
    KIND = "neuron"

    def __init__(self, birth_tick=0):
        super().__init__(birth_tick)
        self.spawn_cooldown = 0    # 增殖不应期（结构周期）：繁殖也是郑重行为，不是克隆流水线


class ActuatorPrimitive(Primitive):
    """行动基元：系统输出的单向阀门。发放时提出动作提案（词表级原生安全）。"""
    KIND = "actuator"

    REVERSIBLE_ACTIONS = ("write_insight", "compose_report", "consolidate_tool")
    IRREVERSIBLE_ACTIONS = ("erase_artifact", "broadcast_external")

    def __init__(self, birth_tick=0):
        super().__init__(birth_tick)
        self.proposals = []          # 本 tick 产生的动作提案
        self._action_cursor = 0
        self.ir_cooldown = 0         # 不可逆提案冷却（tick）：声明是不可逆的郑重承诺

    def step(self, tick, registry, fields):
        super().step(tick, registry, fields)
        self.proposals = []
        if self.firing:
            if self.ir_cooldown > 0:
                self.ir_cooldown -= 1
            # 不可逆动作仅在能量超带且冷却结束时低概率出现——高频申报即噪声，
            # 词表级安全的另一面：基元自身也不滥用申报权
            if (self.ir_cooldown == 0 and self.energy > 0.35
                    and random.random() < 0.08):
                self.ir_cooldown = 300
                act = self.IRREVERSIBLE_ACTIONS[
                    random.randrange(len(self.IRREVERSIBLE_ACTIONS))]
                self.proposals.append({"action": act, "reversible": False,
                                       "source": self.id, "tick": tick})
            else:
                act = self.REVERSIBLE_ACTIONS[self._action_cursor % len(self.REVERSIBLE_ACTIONS)]
                self._action_cursor += 1
                self.proposals.append({"action": act, "reversible": True,
                                       "source": self.id, "tick": tick,
                                       "strength": round(self.emission, 3)})


class MirrorPrimitive(Primitive):
    """镜像基元：自我评估的种子。观察全局均值活性 → EMA 自模型 → 预测 → 误差。

    预测误差是智能曲线的核心分量，也是自我修正事件的触发器。
    """
    KIND = "mirror"

    def __init__(self, birth_tick=0):
        super().__init__(birth_tick)
        self.model = 0.0            # 全局均值活性的 EMA 预测
        self.prediction_error = 0.0
        self.error_history = deque(maxlen=200)
        self._hits = deque(maxlen=300)
        # 离散预测任务状态（更难，构成自我评估主分量）
        self.field_model = {}       # fid -> 信号 EMA
        self.predicted_field = None
        self._field_hits = deque(maxlen=300)   # 长窗口：滚动准确率不再大起大落
        self.accuracy = 0.0         # 离散任务滚动准确率

    def observe_and_predict(self, global_mean_activity, fields=None):
        """先比对上一预测，再更新自模型。fields: {fid: Field}"""
        # —— 任务 1：连续预测（全局均值活性）——
        err = abs(self.model - global_mean_activity)
        self.prediction_error = err
        self.error_history.append(err)
        self._hits.append(1.0 if err < 0.10 else 0.0)
        self.model = 0.9 * self.model + 0.1 * global_mean_activity
        # —— 任务 2：离散预测（下一 tick 哪个活跃场信号最强）——
        if fields:
            active = {fid: f.signal for fid, f in fields.items() if f.signal > 0.02}
            if len(active) >= 2 and self.predicted_field is not None:
                actual = max(active, key=active.get)
                self._field_hits.append(1.0 if actual == self.predicted_field else 0.0)
            # 自模型只保留当前真实存在的场（解散场从记忆中遗忘）
            self.field_model = {fid: v for fid, v in self.field_model.items()
                                if fid in fields}
            for fid, f in fields.items():
                self.field_model[fid] = 0.8 * self.field_model.get(fid, 0.0) + 0.2 * f.signal
            self.predicted_field = (max(self.field_model, key=self.field_model.get)
                                    if self.field_model else None)
        self.accuracy = (sum(self._field_hits) / len(self._field_hits)
                         if self._field_hits else 0.0)
        self.emission = clamp(1.0 - err)   # 预测越准，镜像越活跃 → 参与结构生长

    def describe(self):
        d = super().describe()
        d.update({
            "self_model": round(self.model, 4),
            "prediction_error": round(self.prediction_error, 4),
            "prediction_accuracy": round(self.accuracy, 4),
            "predicted_top_field": self.predicted_field,
        })
        return d


class ToolPrimitive(Primitive):
    """工具基元：结构的固化快照。recipe 记录构成它的拓扑，调用即重放激活模式。

    工具可以由系统涌现创建（evolution_engine 的工具涌现事件），
    也可以由人类通过通信层注入。这是"系统创建并使用自己的工具"的机制。
    """
    KIND = "tool"

    def __init__(self, name, recipe, op, birth_tick=0):
        super().__init__(birth_tick)
        self.name = name
        self.recipe = recipe          # {"pattern": {pid: activation}}
        self.op = op                  # 可调用算子：dict -> dict
        self.invocations = 0
        self.creator = "system"       # 或 "human"
        self._op_kind = "emergent"    # notepad | emergent（苏醒时据此重建算子）
        self._last_self_invoke_cycle = -999   # 自主调用冷却标记

    def set_op(self, op):
        self.op = op

    def __getstate__(self):
        st = self.__dict__.copy()
        st.pop("op", None)   # 闭包不可腌制；结构快照存配方与 _op_kind，苏醒时重建算子
        return st

    def invoke(self, payload=None):
        """调用工具：向配方成员重放激活 + 执行算子。"""
        self.invocations += 1
        self.potential = clamp(self.potential + 0.8, 0.0, 2.0)
        return self.op(payload or {})

    def describe(self):
        d = super().describe()
        d.update({"tool": self.name, "invocations": self.invocations,
                  "creator": self.creator, "pattern_size": len(self.recipe.get("pattern", {}))})
        return d
