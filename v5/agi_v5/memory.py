"""#2 分层记忆系统：工作记忆 / 情景记忆 / 语义记忆 / 程序记忆。

蓝图映射（对话结论）：
  - 工作记忆   = 当前的“意识缓冲”（对应 LLM 的上下文窗口角色）
  - 情景记忆   = 可回放的具体经历：“上次遇到这个问题我是怎么解决的”
  - 语义记忆   = 压缩后的知识结构，激活随时间衰减、访问被强化
  - 程序记忆   = 内化技能，成功率用 Beta 分布维护，Thompson 采样选择
"""
from __future__ import annotations

import math

import numpy as np


class WorkingMemory:
    """容量有限的当前意识缓冲；按显著性淘汰而非纯 FIFO。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = []  # [(salience, t, payload)]

    def push(self, payload, salience: float = 1.0, t: float | None = None):
        t = len(self.items) if t is None else t
        self.items.append((float(salience), t, payload))
        if len(self.items) > self.capacity:
            min_i = min(range(len(self.items)), key=lambda i: self.items[i][0])
            return self.items.pop(min_i)  # 返回被淘汰者（可转入情景层）
        return None

    def recent(self, k: int | None = None):
        k = len(self.items) if k is None else k
        return [p for _, _, p in self.items[-k:]]

    def clear(self):
        self.items.clear()

    def __len__(self):
        return len(self.items)


class EpisodicMemory:
    """带时间戳的情节库；检索得分 = 0.7*余弦相似度 + 0.3*近因性。"""

    def __init__(self, capacity: int, halflife_steps: float = 200.0):
        self.capacity = capacity
        self.halflife = halflife_steps
        self.vecs: list[np.ndarray] = []
        self.meta: list[dict] = []
        self.t: list[int] = []

    def remember(self, vec, meta: dict):
        self.vecs.append(np.asarray(vec, dtype=np.float64))
        self.meta.append(dict(meta))
        self.t.append(int(meta.get("step", len(self.t))))
        if len(self.vecs) > self.capacity:
            self.vecs.pop(0)
            self.meta.pop(0)
            self.t.pop(0)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def retrieve(self, query_vec, k: int = 3, cur_step: int | None = None):
        if not self.vecs:
            return []
        q = self._norm(np.asarray(query_vec, dtype=np.float64))
        V = np.stack([self._norm(v) for v in self.vecs])
        cos = V @ q
        cur_step = len(self.t) if cur_step is None else cur_step
        recency = np.exp(-(cur_step - np.arange(len(self.t))) * (math.log(2.0) / self.halflife))
        score = 0.7 * cos + 0.3 * recency
        idx = np.argsort(-score)[: max(k, 0)]
        return [
            {"meta": self.meta[i], "score": float(score[i]), "cos": float(cos[i])}
            for i in idx
        ]

    def __len__(self):
        return len(self.vecs)


class SemanticMemory:
    """概念节点 + 共现边；激活随时间衰减、访问被强化。

    说明（诚实标注）：v5 中语义记忆是知识结构的可观测支架——
    它参与日志、衰减与联想检索，尚未反哺决策回路（v6 语言接口的挂载点）。
    """

    class Node:
        __slots__ = ("activation", "count")

        def __init__(self):
            self.activation = 0.0
            self.count = 0

    def __init__(self, decay: float = 0.995):
        self.decay = decay
        self.nodes: dict[str, SemanticMemory.Node] = {}
        self.edges: dict[str, dict[str, float]] = {}

    def touch(self, name: str, strength: float = 1.0) -> float:
        node = self.nodes.setdefault(name, self.Node())
        node.activation = min(node.activation + strength, 10.0)
        node.count += 1
        return node.activation

    def co_activate(self, a: str, b: str, strength: float = 1.0):
        self.touch(a, strength)
        self.touch(b, strength)
        ea = self.edges.setdefault(a, {})
        eb = self.edges.setdefault(b, {})
        ea[b] = min(ea.get(b, 0.0) + 0.5 * strength, 10.0)
        eb[a] = min(eb.get(a, 0.0) + 0.5 * strength, 10.0)

    def decay_all(self):
        for n in self.nodes.values():
            n.activation *= self.decay
        for m in self.edges.values():
            for k in m:
                m[k] *= self.decay

    def related(self, name: str, topk: int = 5):
        return sorted(self.edges.get(name, {}).items(), key=lambda kv: -kv[1])[:topk]

    def salient(self, topk: int = 8):
        items = sorted(self.nodes.items(), key=lambda kv: -kv[1].activation)
        return [(name, round(n.activation, 3)) for name, n in items[:topk]]

    def __len__(self):
        return len(self.nodes)


class ProceduralMemory:
    """技能库：Thompson 采样选择，成功率用 Beta(alpha, beta) 维护。"""

    class Skill:
        __slots__ = ("name", "fn", "context", "alpha", "beta", "uses")

        def __init__(self, name, fn, context="*"):
            self.name = name
            self.fn = fn          # callable(obs) -> action 或 None
            self.context = context
            self.alpha = 1.0      # 先验 Beta(1,1) = 无知均匀
            self.beta = 1.0
            self.uses = 0

        @property
        def mean(self) -> float:
            return self.alpha / (self.alpha + self.beta)

    def __init__(self, rng):
        self.rng = rng
        self.skills: dict[str, ProceduralMemory.Skill] = {}

    def register(self, name: str, fn, context: str = "*"):
        self.skills[name] = self.Skill(name, fn, context)

    def select(self, context: str | None = None, explore: bool = True):
        cands = [s for s in self.skills.values() if s.context in ("*", context)]
        if not cands:
            return None
        best, best_sample = None, -np.inf
        for s in cands:
            sample = float(self.rng.beta(s.alpha, s.beta)) if explore else s.mean
            if sample > best_sample:
                best, best_sample = s, sample
        return best

    def report(self, name: str, success: bool):
        s = self.skills.get(name)
        if s is not None:
            s.uses += 1
            if success:
                s.alpha += 1.0
            else:
                s.beta += 1.0
