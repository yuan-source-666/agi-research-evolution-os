"""全局配置：所有子系统的超参数集中于此。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AGIConfig:
    # --- 核心维度（须与环境观测布局一致）---
    obs_dim: int = 14          # NonstationaryGridWorld.OBS_DIM
    action_dim: int = 4
    hidden: int = 48

    # --- #1 持续学习 ---
    lr: float = 3e-3
    gamma: float = 0.95
    replay_capacity: int = 4000
    batch_size: int = 32
    sleep_interval: int = 120  # 每隔多少全局步进入一次巩固（sleep）
    anchor_lam: float = 0.02   # L2-SP 锚定强度（稳定性-可塑性折中）

    # --- #2 记忆 ---
    working_capacity: int = 12
    episodic_capacity: int = 3000
    semantic_decay: float = 0.995

    # --- #4 元认知 ---
    confidence_target: float = 0.75
    max_deliberation_iters: int = 8
    rumination_guard: float = 1e-3   # 反刍闸门：无改善即停
    novelty_weight: float = 0.5

    # --- #5 动机 ---
    curiosity_clip: float = 2.0
    goal_timeout: int = 60

    # --- 环境 ---
    grid_size: int = 8
    phase_length: int = 150    # 非平稳切换周期（步）
    n_phases: int = 4

    seed: int = 7
