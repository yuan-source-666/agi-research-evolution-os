"""非平稳 GridWorld：周期性重排墙壁与目标，用于检验持续适应能力。

设计意图：世界会变（非平稳性）——这正是“训练与部署边界消失”的
最小可检验场景。相位切换后，智能体必须靠持续学习重新适应。

观测布局（obs_dim = 14）：
  [0] x 归一化        [1] y 归一化
  [2] 目标相对 dx     [3] 目标相对 dy
  [4] 归一化曼哈顿距离   <- WorldModel.DIST_IDX
  [5:14] 3x3 局部墙壁视图（出界视为墙）

奖励：到达目标 +1；撞墙/出界 -0.05；每步 -0.01。
动作：0:+x  1:-x  2:+y  3:-y
"""
from __future__ import annotations

import numpy as np


class NonstationaryGridWorld:
    OBS_DIM = 14
    DIST_IDX = 4
    ACTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(self, size: int = 8, n_phases: int = 4, phase_length: int = 150,
                 seed: int = 0, wall_ratio: float = 0.18):
        self.size = size
        self.n_phases = n_phases
        self.phase_length = phase_length
        self.wall_ratio = wall_ratio
        self.rng = np.random.default_rng(seed)
        self.phase = 0
        self.steps_in_phase = 0
        self.walls: set[tuple[int, int]] = set()
        self.goal: tuple[int, int] = (size - 1, size - 1)
        self.pos: tuple[int, int] = (0, 0)
        self.episode_return = 0.0
        self.shifts_seen = 0
        self._new_phase()

    # ---- 相位管理（非平稳性的来源）----
    def _new_phase(self):
        n = self.size
        while True:
            mask = self.rng.random((n, n)) < self.wall_ratio
            start = (0, 0)
            gx, gy = self.rng.integers(n // 2, n, size=2)
            goal = (int(gx), int(gy))
            mask[start] = False
            mask[goal] = False
            free_neighbor = any(
                0 <= start[0] + dx < n and 0 <= start[1] + dy < n
                and not mask[start[0] + dx, start[1] + dy]
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            if free_neighbor:
                break
        self.walls = {(int(x), int(y)) for x, y in zip(*np.where(mask))}
        self.goal = goal
        self.pos = start

    def tick_phase(self):
        """每个全局步调用一次；跨过周期阈值则切换相位。"""
        self.steps_in_phase += 1
        if self.steps_in_phase >= self.phase_length:
            self.steps_in_phase = 0
            self.phase = (self.phase + 1) % self.n_phases
            prev_goal = self.goal
            self._new_phase()
            self.shifts_seen += 1
            return {"shifted": True, "prev_goal": prev_goal}
        return {"shifted": False}

    # ---- 基本接口 ----
    def reset_pos(self):
        self.pos = (0, 0)
        self.episode_return = 0.0

    def in_bounds_free(self, cell) -> bool:
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size and cell not in self.walls

    def step(self, action: int):
        gx, gy = self.goal
        prev_dist = abs(gx - self.pos[0]) + abs(gy - self.pos[1])
        dx, dy = self.ACTIONS[int(action)]
        cand = (self.pos[0] + dx, self.pos[1] + dy)
        reward = -0.01
        bumped = False
        if not self.in_bounds_free(cand):
            bumped = True
            reward -= 0.05
        else:
            self.pos = cand
        new_dist = abs(gx - self.pos[0]) + abs(gy - self.pos[1])
        # 势能塑形（potential-based shaping）：靠近目标有微小正奖励，
        # 让稀疏目标在有限步数内可学（不改变最优策略，Ng et al. 1999）。
        reward += 0.05 * (prev_dist - new_dist)
        reached = self.pos == self.goal
        if reached:
            reward += 1.0
        self.episode_return += reward
        return self.observe(), float(reward), {"reached": reached, "bumped": bumped}

    def local_view(self) -> np.ndarray:
        n = self.size
        view = np.zeros((3, 3))
        px, py = self.pos
        for i in range(-1, 2):
            for j in range(-1, 2):
                x, y = px + i, py + j
                if not (0 <= x < n and 0 <= y < n):
                    view[i + 1, j + 1] = 1.0  # 出界视为墙
                elif (x, y) in self.walls:
                    view[i + 1, j + 1] = 1.0
        return view.flatten()

    def observe(self) -> np.ndarray:
        n = self.size
        px, py = self.pos
        gx, gy = self.goal
        dist = abs(gx - px) + abs(gy - py)
        return np.array(
            [
                px / (n - 1),
                py / (n - 1),
                (gx - px) / (n - 1),
                (gy - py) / (n - 1),
                dist / (2 * (n - 1)),
                *self.local_view(),
            ],
            dtype=np.float64,
        )
