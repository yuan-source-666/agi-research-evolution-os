"""#3 内生世界模型：预测 + 反事实模拟（想象）+ CEM 规划。

蓝图映射（对话结论）：
  “规划的本质是在世界模型里预演，而不是在现实里碰壁后修正。”
  本模块在学到的动力学模型上做类别交叉熵方法（CEM）搜索，
  目标 = 预测的“到目标距离”下降 - 惊讶惩罚（避开模型不信任的区域）。
"""
from __future__ import annotations

import numpy as np


class WorldModel:
    def __init__(self, net, obs_dim: int, action_dim: int, dist_idx: int, rng,
                 surprise_threshold: float = 0.35):
        self.net = net
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dist_idx = dist_idx          # obs 中“归一化距离”特征下标
        self.rng = rng
        self.surprise_threshold = surprise_threshold

    # ---- 基础预测 ----
    def _input(self, obs, action) -> np.ndarray:
        onehot = np.zeros(self.action_dim)
        onehot[int(action)] = 1.0
        return np.concatenate([np.asarray(obs, dtype=np.float64), onehot])[None]

    def predict(self, obs, action) -> np.ndarray:
        return self.net.predict(self._input(obs, action))[0]

    def surprise(self, obs, action, next_obs) -> float:
        pred = self.predict(obs, action)
        return float(np.mean((pred - np.asarray(next_obs)) ** 2))

    def imagine_rollout(self, obs0, actions):
        """反事实模拟：在模型内预演一条动作序列（不在现实里试错）。"""
        s = np.asarray(obs0, dtype=np.float64)
        preds = []
        for a in actions:
            s = self.predict(s, int(a))
            preds.append(s.copy())
        return preds

    # ---- 规划：类别 CEM ----
    def plan_cem(self, obs0, horizon: int = 3, pop: int = 16, iters: int = 2,
                 elite_frac: float = 0.35):
        """在想象空间中搜索动作序列。

        返回 (best_first_action, diagnostics)。评分 =
        Σ(预测距离下降) - 0.5*Σ(逐步自洽性误差)。
        """
        obs0 = np.asarray(obs0, dtype=np.float64)
        probs = np.ones((horizon, self.action_dim)) / self.action_dim
        n_elite = max(4, int(pop * elite_frac))
        best_seq = None
        best_score = -np.inf
        elite_surprise_mean = 0.0

        for _ in range(iters):
            seqs = np.zeros((pop, horizon, self.action_dim))
            for t in range(horizon):
                idx = np.array([self.rng.choice(self.action_dim, p=probs[t]) for _ in range(pop)])
                seqs[np.arange(pop), t, idx] = 1.0

            scores = np.zeros(pop)
            surprises = np.zeros(pop)
            for i in range(pop):
                s = obs0
                d_prev = s[self.dist_idx]
                dist_gain, surp = 0.0, 0.0
                for t in range(horizon):
                    a = int(np.argmax(seqs[i, t]))
                    nxt = self.predict(s, a)
                    dist_gain += float(d_prev - nxt[self.dist_idx])
                    surp += float(np.mean((nxt - s) ** 2))
                    d_prev = nxt[self.dist_idx]
                    s = nxt
                scores[i] = dist_gain - 0.5 * surp
                surprises[i] = surp

            order = np.argsort(-scores)
            elite = seqs[order[:n_elite]]
            if scores[order[0]] > best_score:
                best_score = float(scores[order[0]])
                best_seq = elite[0].copy()
            elite_surprise_mean = float(surprises[order[:n_elite]].mean())

            probs = elite.mean(axis=0) + 1e-3
            probs /= probs.sum(axis=1, keepdims=True)

        if best_seq is None:
            first_action = int(self.rng.integers(self.action_dim))
            best_score = -np.inf
        else:
            first_action = int(np.argmax(best_seq[0]))
        diag = {"cem_score": best_score, "cem_surprise": elite_surprise_mean}
        return first_action, diag
