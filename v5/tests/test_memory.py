import numpy as np

from agi_v5.memory import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory


def test_working_memory_evicts_lowest_salience():
    wm = WorkingMemory(capacity=3)
    wm.push({"id": "low"}, salience=0.1)
    wm.push({"id": "high1"}, salience=0.9)
    wm.push({"id": "mid"}, salience=0.5)
    dropped = wm.push({"id": "high2"}, salience=0.8)
    assert len(wm) == 3
    assert dropped is not None and dropped[2]["id"] == "low"


def test_episodic_retrieval_prefers_similar_and_recent():
    em = EpisodicMemory(100)
    base = np.zeros(6)
    em.remember(base + np.array([1, 0, 0, 0, 0, 0]), {"tag": "A", "step": 0})
    em.remember(base + np.array([0, 0, 0, 0, 0, 0.1]), {"tag": "B", "step": 5})
    hits = em.retrieve(base + np.array([0, 0, 0, 0, 0, 0.09]))
    assert hits and hits[0]["meta"]["tag"] == "B"


def test_semantic_decay_and_association():
    sm = SemanticMemory(decay=0.9)
    sm.co_activate("cellA", "goal_reached")
    a0 = sm.nodes["cellA"].activation
    for _ in range(20):
        sm.decay_all()
    assert sm.nodes["cellA"].activation < a0 * 0.3
    related = dict(sm.related("cellA"))
    assert "goal_reached" in related


def test_procedural_thompson_selection_and_stats():
    rng = np.random.default_rng(3)
    pm = ProceduralMemory(rng)
    pm.register("always_right", lambda obs: 0)
    s = pm.select()
    assert s is not None and s.name == "always_right"
    for _ in range(10):
        pm.report("always_right", success=True)
    assert pm.skills["always_right"].mean > 0.7
    assert callable(pm.skills["always_right"].fn)
