from agi_v5.config import AGIConfig
from agi_v5.core import Agent
from agi_v5.envs import NonstationaryGridWorld


def test_agent_runs_end_to_end():
    cfg = AGIConfig()
    cfg.phase_length = 30
    cfg.sleep_interval = 25
    agent = Agent(cfg=cfg, seed=11)
    env = NonstationaryGridWorld(size=cfg.grid_size, n_phases=cfg.n_phases,
                                 phase_length=cfg.phase_length, seed=11)
    summary = agent.run(env, steps=80)
    assert len(agent.trace) == 80
    modes = {r["mode"] for r in agent.trace}
    assert modes <= {"fast", "slow"} and modes  # 快慢双系统都被路由过或至少其一
    assert summary["steps"] == 80


def test_agent_learns_continually_world_model_improves():
    """持续学习的本质检验：世界模型对新转移的预测误差应随经验下降，
    且后期行为噪声（撞墙率）不失控。"""
    cfg = AGIConfig()
    cfg.phase_length = 120
    cfg.sleep_interval = 60
    agent = Agent(cfg=cfg, seed=13)
    env = NonstationaryGridWorld(size=cfg.grid_size, n_phases=2,
                                 phase_length=cfg.phase_length, seed=13)

    early = []
    late = []
    for i in range(240):
        env.tick_phase()
        rec = agent.cognitive_cycle(env)
        if i < 60:
            early.append(rec)
        if 180 <= i < 240:
            late.append(rec)

    early_surp = sum(r["surprise"] for r in early) / len(early)
    late_surp = sum(r["surprise"] for r in late) / len(late)
    assert late_surp < early_surp, (
        f"世界模型未从经验中受益: 前60步惊讶={early_surp:.4f}, 后60步={late_surp:.4f}"
    )
    late_bump = sum(1 for r in late if r["bumped"]) / len(late)
    assert late_bump < 0.75, f"后期行为失控, 撞墙率={late_bump:.2%}"
