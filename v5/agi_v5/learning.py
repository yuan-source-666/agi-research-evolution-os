"""#1 持续学习引擎：在线学习 + 优先回放 + 睡眠巩固。

蓝图映射（对话结论）：
  “训练与部署边界消失”——在线阶段保持可塑性；
  周期性 sleep 阶段向睡眠前权重快照做 L2-SP 锚定收缩，保住稳定性，
  对应“清醒时吸收、睡眠时巩固”的双阶段机制。

诚实标注：锚定采用简化的 L2-SP 正则（向上一阶段解收缩）。
真正的 EWC / Synaptic Intelligence 在 v6 路线图上。
"""
from __future__ import annotations

import numpy as np


class NumpyMLP:
    """极简全连接网络：tanh 隐层 + 线性输出，MSE 损失，手写反向传播。"""

    def __init__(self, sizes, rng, lr: float = 3e-3):
        self.sizes = list(sizes)
        self.lr = lr
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(sizes) - 1):
            w = rng.normal(0.0, np.sqrt(2.0 / sizes[i]), size=(sizes[i], sizes[i + 1]))
            self.W.append(w)
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, x):
        acts = [np.asarray(x, dtype=np.float64)]
        h = acts[0]
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i != last:
                h = np.tanh(h)
            acts.append(h)
        return acts

    def predict(self, x) -> np.ndarray:
        return self.forward(x)[-1]

    def train_step(self, x, y, anchor=None, anchor_lam: float = 0.0) -> float:
        """单批 SGD。anchor=(W,b) 时施加 L2-SP 锚定。返回该批 MSE 损失。"""
        acts = self.forward(x)
        out = acts[-1]
        diff = out - y
        loss = float(np.mean(diff ** 2))
        grad = (2.0 / max(y.size, 1)) * diff
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ grad
            gb[i] = grad.sum(axis=0)
            if i > 0:
                grad = (grad @ self.W[i].T) * (1.0 - acts[i] ** 2)  # tanh 导数
        if anchor is not None and anchor_lam > 0:
            for i in range(len(self.W)):
                gW[i] = gW[i] + anchor_lam * (self.W[i] - anchor[0][i])
                gb[i] = gb[i] + anchor_lam * (self.b[i] - anchor[1][i])
        for i in range(len(self.W)):
            self.W[i] -= self.lr * gW[i]
            self.b[i] -= self.lr * gb[i]
        return loss

    def snapshot(self):
        return ([w.copy() for w in self.W], [b.copy() for b in self.b])

    def distance_to(self, snap) -> float:
        d = 0.0
        for i in range(len(self.W)):
            d += float(np.sum((self.W[i] - snap[0][i]) ** 2))
            d += float(np.sum((self.b[i] - snap[1][i]) ** 2))
        return d


class ExperienceReplay:
    """优先级回放池：惊讶度（预测误差）越高越容易被抽中复习。"""

    def __init__(self, capacity: int, rng):
        self.capacity = capacity
        self.rng = rng
        self.obs: list[np.ndarray] = []
        self.act: list[int] = []
        self.rew: list[float] = []
        self.next_obs: list[np.ndarray] = []
        self.done: list[float] = []
        self.prio: list[float] = []

    def __len__(self):
        return len(self.obs)

    def push(self, obs, action: int, reward: float, next_obs, done: bool, priority: float = 1.0):
        if len(self.obs) < self.capacity:
            self.obs.append(obs)
            self.act.append(int(action))
            self.rew.append(float(reward))
            self.next_obs.append(next_obs)
            self.done.append(float(done))
            self.prio.append(float(priority))
        else:
            i = int(np.argmin(self.prio))  # 低价值记忆先让位
            self.obs[i] = obs
            self.act[i] = int(action)
            self.rew[i] = float(reward)
            self.next_obs[i] = next_obs
            self.done[i] = float(done)
            self.prio[i] = float(priority)

    def sample(self, batch: int) -> dict:
        n = len(self.obs)
        idx = self.rng.choice(n, size=min(batch, n), replace=False, p=self._probs())
        return {
            "obs": np.stack([self.obs[i] for i in idx]),
            "act": np.array([self.act[i] for i in idx], dtype=int),
            "rew": np.array([self.rew[i] for i in idx], dtype=np.float64),
            "next_obs": np.stack([self.next_obs[i] for i in idx]),
            "done": np.array([self.done[i] for i in idx], dtype=np.float64),
            "prio": np.array([self.prio[i] for i in idx], dtype=np.float64),
        }

    def _probs(self) -> np.ndarray:
        p = np.asarray(self.prio, dtype=np.float64)
        p = p / p.sum()
        return p


class ContinualLearningEngine:
    """持有世界模型网络与价值网络，负责在线更新与睡眠巩固。

    - 在线（可塑性）：每个新转移立即做一次小批更新；
    - 睡眠（稳定性）：多批重放 + 向睡眠前快照做 L2-SP 锚定。
    """

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        d, a, h = cfg.obs_dim, cfg.action_dim, cfg.hidden
        # 世界模型：predict(next_obs | obs ⊕ onehot(action))
        self.world_net = NumpyMLP([d + a, h, h, d], rng, lr=cfg.lr)
        # 价值网络（System-1 摊销策略载体）：Q(obs) -> [a_dim]
        self.q_net = NumpyMLP([d, h, h, a], rng, lr=cfg.lr)
        self.replay = ExperienceReplay(cfg.replay_capacity, rng)
        self.gamma = cfg.gamma
        self.anchor_lam = cfg.anchor_lam
        self.world_anchor = None
        self.q_anchor = None
        self.stats = {"online_steps": 0, "sleeps": 0, "world_loss": None, "q_loss": None}

    # ---- 内部工具 ----
    def _onehot(self, actions) -> np.ndarray:
        out = np.zeros((len(actions), self.cfg.action_dim))
        out[np.arange(len(actions)), np.asarray(actions, dtype=int)] = 1.0
        return out

    def _build_batch(self, s: dict):
        inp = np.concatenate([s["obs"], self._onehot(s["act"])], axis=1)
        q_next = self.q_net.predict(s["next_obs"]).max(axis=1)
        target_q = s["rew"] + self.gamma * (1.0 - s["done"]) * q_next
        return inp, s["next_obs"], s["act"], target_q.reshape(-1, 1)

    def _train_on_batch(self, anchors: bool) -> tuple[float, float]:
        inp, next_obs_b, act_b, target_q = self._build_batch(self.replay.sample(self.cfg.batch_size))
        lw = self.world_net.train_step(
            inp, next_obs_b,
            anchor=self.world_anchor if anchors else None,
            anchor_lam=self.anchor_lam if anchors else 0.0,
        )
        obs_b = inp[:, : self.cfg.obs_dim]
        qs = self.q_net.predict(obs_b)
        qs[np.arange(len(act_b)), act_b] = target_q[:, 0]
        lq = self.q_net.train_step(
            obs_b, qs,
            anchor=self.q_anchor if anchors else None,
            anchor_lam=self.anchor_lam if anchors else 0.0,
        )
        self.stats["world_loss"] = lw
        self.stats["q_loss"] = lq
        return lw, lq

    # ---- 公开接口 ----
    def observe(self, obs, action: int, reward: float, next_obs, done: bool) -> float:
        """吸收一个新转移；返回更新前世界模型的惊讶度（预测误差）。"""
        onehot = self._onehot([action])[0]
        inp = np.concatenate([obs, onehot])[None]
        pred = self.world_net.predict(inp)[0]
        surprise = float(np.mean((pred - next_obs) ** 2))

        self.replay.push(obs, action, reward, next_obs, done, priority=1.0 + 10.0 * surprise)
        if len(self.replay) >= max(8, self.cfg.batch_size // 4):
            self._train_on_batch(anchors=False)
        self.stats["online_steps"] += 1
        return surprise

    def sleep(self, n_batches: int = 20) -> dict:
        """巩固阶段：向睡眠前快照锚定的重放训练（stability-plasticity 折中）。"""
        if len(self.replay) < self.cfg.batch_size:
            return {"skipped": True}
        self.world_anchor = self.world_net.snapshot()
        self.q_anchor = self.q_net.snapshot()
        lw = lq = 0.0
        for _ in range(n_batches):
            a, b = self._train_on_batch(anchors=True)
            lw += a
            lq += b
        self.stats["sleeps"] += 1
        return {"sleeps": self.stats["sleeps"], "world_loss": lw / n_batches, "q_loss": lq / n_batches}
