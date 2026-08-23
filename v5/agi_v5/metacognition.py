"""#4 元认知：信心估计 + 计算调度器。

蓝图映射（对话结论）：
  “知道自己知道什么、为难题分配更多计算、察觉自己正在犯错，
   以及——最被低估的——知道何时停止。不会停下来的思考不是深思，是反刍。”

三重停车条件：
  1. 信心达标（confidence）
  2. 计算预算耗尽（budget）
  3. 反刍闸门（rumination-guard）：连续迭代无改善即停
"""
from __future__ import annotations

import numpy as np


def q_confidence(q_values) -> float:
    """softmax 概率的 top1-top2 间距 ∈ [0,1]，作为 System-1 的信心。"""
    q = np.asarray(q_values, dtype=np.float64)
    e = np.exp(q - q.max())
    p = e / e.sum()
    sp = np.sort(p)[::-1]
    margin = float(sp[0] - sp[1]) if len(sp) > 1 else 1.0
    return min(max(margin, 0.0), 1.0)


class MetacognitiveScheduler:
    def __init__(self, confidence_target: float = 0.75, max_iters: int = 8,
                 rumination_guard: float = 1e-3, novelty_weight: float = 0.5):
        self.confidence_target = confidence_target
        self.max_iters = max_iters
        self.rumination_guard = rumination_guard
        self.novelty_weight = novelty_weight
        self.compute_spent = {"fast": 0, "slow": 0}

    def route(self, fast_confidence: float, novelty: float) -> str:
        """快慢分流：新颖度高或直觉信心不足 → 深思。

        有效阈值 = confidence_target + novelty_weight * min(novelty, 0.5)
        越陌生的局面，System-1 的自信越不可信，
        快速通道的门槛就越高——越值得多想一会儿。
        """
        threshold = self.confidence_target + self.novelty_weight * min(novelty, 0.5)
        threshold = min(threshold, 0.99)
        mode = "fast" if fast_confidence >= threshold else "slow"
        self.compute_spent[mode] += 1
        return mode

    def should_continue_thinking(self, iters: int, best_score_history: list[float],
                                 confidence: float):
        """返回 (是否继续, 停止原因)。"""
        if confidence >= self.confidence_target:
            return False, "confidence"
        if iters >= self.max_iters:
            return False, "budget"
        if len(best_score_history) >= 2:
            improvement = best_score_history[-1] - best_score_history[-2]
            if improvement < self.rumination_guard:
                return False, "rumination-guard"
        return True, None
