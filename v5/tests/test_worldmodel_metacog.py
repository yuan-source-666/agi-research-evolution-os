import numpy as np

from agi_v5.config import AGIConfig
from agi_v5.learning import NumpyMLP
from agi_v5.metacognition import MetacognitiveScheduler, q_confidence
from agi_v5.world_model import WorldModel


def test_world_model_predict_shapes_and_surprise_zero_for_consistent():
    rng = np.random.default_rng(4)
    cfg = AGIConfig()
    net = NumpyMLP([cfg.obs_dim + cfg.action_dim, 32, cfg.obs_dim], rng)
    wm = WorldModel(net, cfg.obs_dim, cfg.action_dim, dist_idx=4, rng=rng)
    obs = np.abs(rng.normal(size=cfg.obs_dim))
    pred = wm.predict(obs, 2)
    assert pred.shape == (cfg.obs_dim,)
    assert wm.surprise(obs, 2, pred) < 1e-12


def test_plan_returns_valid_action():
    rng = np.random.default_rng(5)
    cfg = AGIConfig()
    net = NumpyMLP([cfg.obs_dim + cfg.action_dim, 32, cfg.obs_dim], rng)
    wm = WorldModel(net, cfg.obs_dim, cfg.action_dim, dist_idx=4, rng=rng)
    obs = np.abs(rng.normal(size=cfg.obs_dim))
    obs[4] = 0.5
    action, diag = wm.plan_cem(obs)
    assert 0 <= action < cfg.action_dim
    assert "cem_score" in diag


def test_q_confidence_margin():
    assert q_confidence([10.0, 0.0, 0.0, 0.0]) > 0.9
    assert q_confidence([1.0, 1.0, 1.0, 1.0]) < 0.01


def test_scheduler_routes_by_novelty():
    sch = MetacognitiveScheduler(confidence_target=0.75)
    assert sch.route(0.9, novelty=0.0) == "fast"
    assert sch.route(0.3, novelty=0.4) == "slow"


def test_scheduler_stop_conditions():
    sch = MetacognitiveScheduler(confidence_target=0.75, max_iters=8,
                                 rumination_guard=1e-3)
    cont, reason = sch.should_continue_thinking(1, [0.1], 0.95)
    assert not cont and reason == "confidence"
    cont, reason = sch.should_continue_thinking(9, [0.1], 0.3)
    assert not cont and reason == "budget"
    flat = [0.5, 0.5]
    cont, reason = sch.should_continue_thinking(3, flat, 0.3)
    assert not cont and reason == "rumination-guard"
