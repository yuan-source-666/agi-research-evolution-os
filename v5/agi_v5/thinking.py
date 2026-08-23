"""快慢双系统思考循环（原则二 + 原则三）。

蓝图映射（对话结论）：
  思考 = 可延长的计算时间 × 可信的验证闭环 × 知道何时停止的元认知。
  System 1（FastPath）：程序技能优先，摊销价值网络兜底——直觉草案；
  System 2（ThinkLoop）：在世界模型内生成候选 → 验证打分 → 迭代修正，
  由元认知调度器踩刹车（信心达标 / 预算耗尽 / 反刍闸门）。
"""
from __future__ import annotations

import numpy as np


class FastPath:
    """System 1：摊销计算——把常见问题的解法压进权重，一次前向即答案。"""

    def __init__(self, procedural_memory, engine, cfg):
        self.pm = procedural_memory
        self.engine = engine
        self.cfg = cfg

    def propose(self, obs) -> dict:
        skill = self.pm.select()
        if skill is not None and skill.mean > 0.45 and callable(skill.fn):
            try:
                action = int(skill.fn(obs))
            except Exception:
                action = None
            if action is not None and 0 <= action < self.cfg.action_dim:
                return {"action": action, "source": f"skill:{skill.name}",
                        "skill": skill.name, "confidence": skill.mean}
        q = self.engine.q_net.predict(np.asarray(obs)[None])[0]
        return {"action": int(np.argmax(q)), "source": "q-net", "q": q.tolist()}


class ThinkLoop:
    """System 2：候选生成（CEM 规划 / 启发式 / 探索探针）
    → 三重验证打分（模型内收益 - 惊讶惩罚 - 情景失败惩罚）
    → 元认知刹车。"""

    def __init__(self, world_model, metacog, episodic, cfg, dist_idx, heuristic=None):
        self.wm = world_model
        self.meta = metacog
        self.episodic = episodic
        self.cfg = cfg
        self.dist_idx = dist_idx
        self.heuristic = heuristic  # fn(obs) -> action 或 None

    def _failure_penalty(self, obs, action: int) -> float:
        """情景校验：相似局面下同一动作吃过亏吗？（记忆反哺决策）"""
        hits = self.episodic.retrieve(obs, k=2)
        penalty = 0.0
        for h in hits:
            m = h["meta"]
            if m.get("action") == int(action) and m.get("reward", 0.0) < 0:
                penalty += 0.25 * h["score"]
        return penalty

    def deliberate(self, obs, novelty: float = 0.0):
        obs = np.asarray(obs, dtype=np.float64)
        trace = {"iterations": [], "stop_reason": None}
        best_action, best_name, best_score = None, None, -np.inf
        history: list[float] = []
        horizon = 3
        it = 0

        while True:
            it += 1
            candidates: list[tuple[str, int]] = []
            act, diag = self.wm.plan_cem(obs, horizon=horizon, pop=12 + 6 * it, iters=2)
            candidates.append(("plan:cem", int(act)))
            if self.heuristic is not None:
                ha = int(self.heuristic(obs))
                if it > 1:
                    ha = int((ha + it) % self.cfg.action_dim)  # 扰动制造新候选
                candidates.append(("heuristic", ha))
            candidates.append(("random-probe", int(self.wm.rng.integers(self.cfg.action_dim))))

            improved_this_iter = False
            for name, a in candidates:
                nxt = self.wm.predict(obs, a)
                gain = float(obs[self.dist_idx] - nxt[self.dist_idx])
                surp = self.wm.surprise(obs, a, nxt)
                fail = self._failure_penalty(obs, a)
                score = gain - 0.5 * surp - fail
                if score > best_score + 1e-12:
                    best_score, best_action, best_name = score, a, name
                    improved_this_iter = True

            conf = float(np.clip(0.5 + 0.5 * best_score, 0.0, 1.0))
            history.append(best_score)
            trace["iterations"].append({
                "best": best_name,
                "score": round(float(best_score), 4),
                "conf": round(conf, 3),
            })

            cont, reason = self.meta.should_continue_thinking(it, history, conf)
            if not cont:
                trace["stop_reason"] = reason
                break
            if not improved_this_iter:
                trace["stop_reason"] = "no-improvement"
                break

        trace["action"] = best_action
        trace["source"] = best_name
        trace["score"] = float(best_score) if best_action is not None else 0.0
        return best_action, trace
