"""Agent 认知周期编排：感知 → 动机 → 快/慢决策 → 行动 → 学习 → 记忆 → 睡眠。

这是“五件套”的装配车间。单个 cognitive_cycle 展示一次完整的心智事件：
直觉先给草案，元认知决定要不要深思，行动之后所有子系统同步吸收经验，
并周期性进入睡眠巩固。
"""
from __future__ import annotations

import numpy as np

from .config import AGIConfig
from .envs import NonstationaryGridWorld
from .learning import ContinualLearningEngine
from .memory import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
from .metacognition import MetacognitiveScheduler, q_confidence
from .motivation import CompetenceTracker, CuriosityDrive, GoalSystem
from .thinking import FastPath, ThinkLoop
from .world_model import WorldModel


def greedy_heuristic(obs) -> int:
    """内置启发式技能：朝目标方向走（可能撞墙——正好让学习修正它）。"""
    dx, dy = float(obs[2]), float(obs[3])
    if abs(dx) >= abs(dy):
        return 0 if dx >= 0 else 1
    return 2 if dy >= 0 else 3


class Agent:
    def __init__(self, cfg: AGIConfig | None = None, seed: int = 7):
        self.cfg = cfg or AGIConfig()
        self.rng = np.random.default_rng(seed)

        # 五件套装配
        self.engine = ContinualLearningEngine(self.cfg, self.rng)          # #1
        self.working = WorkingMemory(self.cfg.working_capacity)            # #2
        self.episodic = EpisodicMemory(self.cfg.episodic_capacity)         # #2
        self.semantic = SemanticMemory(decay=self.cfg.semantic_decay)      # #2
        self.pm = ProceduralMemory(self.rng)                               # #2
        self.pm.register("greedy_to_goal", greedy_heuristic, context="*")
        self.world = WorldModel(                                           # #3
            self.engine.world_net, self.cfg.obs_dim, self.cfg.action_dim,
            dist_idx=NonstationaryGridWorld.DIST_IDX, rng=self.rng,
        )
        self.meta = MetacognitiveScheduler(                                # #4
            confidence_target=self.cfg.confidence_target,
            max_iters=self.cfg.max_deliberation_iters,
            rumination_guard=self.cfg.rumination_guard,
            novelty_weight=self.cfg.novelty_weight,
        )
        self.curiosity = CuriosityDrive(clip=self.cfg.curiosity_clip)      # #5
        self.goals = GoalSystem(timeout=self.cfg.goal_timeout,             # #5
                                grid_size=self.cfg.grid_size, rng=self.rng)
        self.competence = CompetenceTracker()                              # #5

        # 双系统
        self.fast = FastPath(self.pm, self.engine, self.cfg)
        self.slow = ThinkLoop(self.world, self.meta, self.episodic, self.cfg,
                              dist_idx=NonstationaryGridWorld.DIST_IDX,
                              heuristic=greedy_heuristic)

        # 状态
        self.step_count = 0
        self.trace: list[dict] = []
        self.visit_counts: dict[tuple[int, int], int] = {}
        self.last_surprise = 0.0
        self.surprise_ema: float | None = None  # 惊讶度运行均值（相对新颖度的基线）

    # ---------- 单步认知周期 ----------
    def cognitive_cycle(self, env: NonstationaryGridWorld) -> dict:
        obs = env.observe()
        self.step_count += 1
        t = self.step_count

        # 1. 感知入工作记忆
        self.working.push({"kind": "obs", "pos": env.pos}, salience=1.0, t=t)

        # 2. 动机：对上一步惊讶度的好奇心内在奖励
        r_intrinsic = self.curiosity.reward(self.last_surprise)
        # 相对新颖度：惊讶度显著高于自身典型水平才算“陌生”。
        # 语义：平稳期 → 0（信任直觉快答）；相位切换等异常 → 尖峰（触发深思）。
        if self.surprise_ema is None:
            self.surprise_ema = self.last_surprise
        self.surprise_ema = 0.95 * self.surprise_ema + 0.05 * self.last_surprise
        novelty = float(np.clip(
            (self.last_surprise - 1.5 * self.surprise_ema) / (self.surprise_ema + 1e-3),
            0.0, 1.0,
        ))

        # 3. System-1 草案与信心估计
        fast_prop = self.fast.propose(obs)
        if str(fast_prop.get("source", "")).startswith("skill"):
            conf_fast = float(fast_prop.get("confidence", 0.5))
        else:
            conf_fast = q_confidence(fast_prop["q"])

        # 4. 元认知路由：快答或深思
        mode = self.meta.route(conf_fast, novelty)
        think_trace = None
        if mode == "fast":
            action, via = fast_prop["action"], fast_prop["source"]
        else:
            action, think_trace = self.slow.deliberate(obs, novelty=novelty)
            action = int(action) if action is not None else fast_prop["action"]
            via = think_trace.get("source")

        # 4.5 ε-贪婪探索（随经验衰减）：世界在变，策略必须保持可修正性
        eps = max(0.03, 0.20 - 0.0005 * self.step_count)
        explored = self.rng.random() < eps
        if explored:
            action = int(self.rng.integers(self.cfg.action_dim))

        # 5. 行动
        next_obs, r_ext, info = env.step(action)

        # 6. 在线学习（#1）；更新前的预测误差成为下一步的新颖度信号
        surprise_next = self.engine.observe(obs, action, r_ext, next_obs, done=False)
        self.last_surprise = surprise_next

        # 7. 分层记忆写入（#2）
        ep_meta = {
            "step": t,
            "pos": tuple(env.pos),
            "action": int(action),
            "reward": float(r_ext),
            "mode": mode,
            "reached": bool(info.get("reached")),
        }
        self.episodic.remember(obs, ep_meta)
        self.visit_counts[env.pos] = self.visit_counts.get(env.pos, 0) + 1
        if info.get("bumped"):
            self.semantic.co_activate(f"cell{env.pos}", "wall_bump")
        if str(via or "").startswith(("skill", "heuristic")):
            self.pm.report("greedy_to_goal",
                           success=(not info.get("bumped")) and r_ext > 0.0)

        # 8. 目标栈管理（#5）
        goal = self.goals.maybe_new_goal(env.pos, self.visit_counts, t,
                                         achieved=bool(info.get("reached")))
        if info.get("reached"):
            self.semantic.co_activate(f"cell{env.pos}", "goal_reached")
            self.goals.push_exploit(env.goal, t)  # 新目标点已知 → 利用

        # 9. 可观测痕迹（元认知的外化日志）
        rec = {
            "step": t,
            "mode": mode,
            "via": via,
            "action": int(action),
            "r_ext": float(r_ext),
            "r_int": float(r_intrinsic),
            "surprise": float(surprise_next),
            "novelty": round(novelty, 3),
            "explored": bool(explored),
            "conf_fast": round(float(conf_fast), 3),
            "goal_kind": goal.kind if goal else None,
            "stop_reason": (think_trace or {}).get("stop_reason"),
            "phase": env.phase,
            "reached": bool(info.get("reached")),
            "bumped": bool(info.get("bumped")),
        }
        self.trace.append(rec)
        return rec

    # ---------- 多步运行 ----------
    def run(self, env: NonstationaryGridWorld, steps: int) -> dict:
        episodes = 0
        cur_ret = 0.0
        for i in range(steps):
            shift = env.tick_phase()
            rec = self.cognitive_cycle(env)
            cur_ret += rec["r_ext"]
            if rec["reached"]:
                self.competence.update(cur_ret)
                episodes += 1
                cur_ret = 0.0
                env.reset_pos()
            if (i + 1) % self.cfg.sleep_interval == 0:
                self.engine.sleep()
                self.semantic.decay_all()
        return {
            "steps": steps,
            "episodes": episodes,
            "sleeps": self.engine.stats["sleeps"],
            "shifts_seen": env.shifts_seen,
        }
