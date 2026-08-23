import numpy as np

from agi_v5.config import AGIConfig
from agi_v5.learning import ContinualLearningEngine, ExperienceReplay, NumpyMLP


def test_mlp_learns_identity():
    rng = np.random.default_rng(0)
    net = NumpyMLP([4, 16, 4], rng, lr=0.05)
    X = rng.normal(size=(256, 4))
    Y = X.copy()
    before = float(np.mean((net.predict(X) - Y) ** 2))
    for _ in range(200):
        idx = rng.choice(256, size=32)
        net.train_step(X[idx], Y[idx])
    after = float(np.mean((net.predict(X) - Y) ** 2))
    assert after < before * 0.5, f"损失未显著下降: {before:.4f} -> {after:.4f}"


def test_replay_priority_sampling_biases_toward_high_priority():
    rng = np.random.default_rng(1)
    rp = ExperienceReplay(100, rng)
    for i in range(100):
        rp.push(np.zeros(4), 0, 0.0, np.zeros(4), False, priority=float(i))
    s = rp.sample(32)
    assert s["obs"].shape[0] == 32
    means = [rp.sample(64)["prio"].mean() for _ in range(20)]
    assert float(np.mean(means)) > 55.0, f"优先级采样未生效: {np.mean(means):.1f}"


def test_sleep_anchor_limits_weight_drift():
    cfg = AGIConfig()
    cfg.batch_size = 8
    rng = np.random.default_rng(2)
    eng = ContinualLearningEngine(cfg, rng)
    for _ in range(64):
        o = rng.normal(size=cfg.obs_dim)
        n2 = rng.normal(size=cfg.obs_dim) * 0.1
        eng.observe(o, int(rng.integers(cfg.action_dim)), 0.0, n2, False)
    snap = eng.world_net.snapshot()
    result = eng.sleep(n_batches=10)
    assert not result.get("skipped", False)
    drift = eng.world_net.distance_to(snap)
    assert drift < 2.0, f"锚定未能限制漂移: {drift:.3f}"
