"""#5 内生动机系统：好奇心 + 目标栈 + 胜任感。

蓝图映射（对话结论）：
  “没有内生目标，再强的系统也只是‘模拟智能行为的镜子’；
   有了它，才是一个真正想要完成事情的智能体。”
  - 好奇心：世界模型预测误差作为内在奖励（随经验整体衰减——新鲜感会过去）
  - 目标栈：利用目标（到达已知奖励点）/ 探索目标（知识缺口 = 最少访问区）
  - 胜任感：分段回报均值的正向增量 → 内在奖励（“我在变强”的感觉）
"""
from __future__ import annotations

from collections import deque

import numpy as np


class CuriosityDrive:
    def __init__(self, clip: float = 2.0, decay: float = 0.999, k: float = 1.0):
        self.clip = clip
        self.decay = decay
        self.k = k
        self.t = 0

    def reward(self, surprise: float) -> float:
        self.t += 1
        base = min(max(float(surprise), 0.0), self.clip)
        return self.k * base * (self.decay ** self.t)


class GoalSystem:
    class Goal:
        __slots__ = ("kind", "target", "born", "note")

        def __init__(self, kind: str, target, born: int, note: str = ""):
            self.kind = kind
            self.target = tuple(target)
            self.born = born
            self.note = note

    def __init__(self, timeout: int = 60, grid_size: int = 8, rng=None):
        self.timeout = timeout
        self.grid = grid_size
        self.rng = rng or np.random.default_rng(0)
        self.stack: list[GoalSystem.Goal] = []

    @property
    def active(self):
        return self.stack[-1] if self.stack else None

    def push_exploit(self, target_cell, step: int):
        """发现已知奖励点 → 切换为利用目标（清空栈）。"""
        self.stack = [self.Goal("exploit", target_cell, step)]

    def maybe_new_goal(self, pos, visit_counts: dict, step: int, achieved: bool = False):
        """过期或达成后生成新子目标：默认转向最少访问的区域（知识缺口）。"""
        active = self.stack[-1] if self.stack else None
        expired = active is None or (step - active.born) > self.timeout
        if achieved:
            self.stack.clear()
            active = None
        if active is None or expired:
            counts = {
                (x, y): visit_counts.get((x, y), 0)
                for x in range(self.grid)
                for y in range(self.grid)
            }
            target = min(counts, key=counts.get)
            goal = self.Goal("explore", target, step, note="knowledge-gap")
            self.stack.append(goal)
            return goal
        return active


class CompetenceTracker:
    """胜任感：最近窗口回报均值的正向增量 → 内在奖励。"""

    def __init__(self, window: int = 50):
        self.window = window
        self.hist: deque[float] = deque(maxlen=window * 4)
        self.prev_mean = 0.0

    def update(self, episode_return: float) -> float:
        self.hist.append(float(episode_return))
        recent = list(self.hist)[-self.window:]
        cur = sum(recent) / len(recent)
        delta = cur - self.prev_mean
        self.prev_mean = cur
        return max(delta, 0.0)
