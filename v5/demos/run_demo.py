"""AGI v5 认知周期端到端演示。

运行： python demos/run_demo.py [steps]
展示：非平稳世界中的持续学习、快慢分流、好奇心驱动、睡眠巩固与记忆成长。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agi_v5.config import AGIConfig             # noqa: E402
from agi_v5.core import Agent                   # noqa: E402
from agi_v5.envs import NonstationaryGridWorld  # noqa: E402


def main(steps=600, seed=11):
    cfg = AGIConfig(seed=seed)
    agent = Agent(cfg=cfg, seed=seed)
    env = NonstationaryGridWorld(size=cfg.grid_size, n_phases=cfg.n_phases,
                                 phase_length=cfg.phase_length, seed=seed)

    t0 = time.time()
    summary = agent.run(env, steps=steps)
    dt = time.time() - t0
    trace = agent.trace

    bar = "=" * 68
    print(bar)
    print(f"AGI v5 认知周期演示   steps={steps}   用时={dt:.1f}s")
    print(f"回合数={summary['episodes']}   睡眠巩固={summary['sleeps']}次   "
          f"相位切换={summary['shifts_seen']}次   在线更新={agent.engine.stats['online_steps']}")
    print(bar)

    W = 100
    header = f"{'窗口':>6} {'相位':>4} {'累计回报':>10} {'慢思考%':>8} {'平均惊讶':>9} {'到达':>4}  停止原因分布"
    print(header)
    for s0 in range(0, steps, W):
        seg = trace[s0:s0 + W]
        ret = sum(r["r_ext"] for r in seg)
        slow_pct = 100.0 * sum(1 for r in seg if r["mode"] == "slow") / len(seg)
        surp = sum(r["surprise"] for r in seg) / len(seg)
        reached = sum(1 for r in seg if r["reached"])
        stops = {}
        for r in seg:
            key = r.get("stop_reason")
            if key:
                stops[key] = stops.get(key, 0) + 1
        print(f"{s0:>6} {seg[0]['phase']:>4} {ret:>10.2f} {slow_pct:>7.1f}% {surp:>9.4f} {reached:>4}  {stops}")

    # 非平稳适应：每个相位的撞墙率（越低 = 切换后再适应越快）
    print()
    print("各相位表现（非平稳切换后的再适应）:")
    for ph in sorted({r["phase"] for r in trace}):
        seg = [r for r in trace if r["phase"] == ph]
        bump = sum(1 for r in seg if r["bumped"]) / len(seg)
        reach = sum(1 for r in seg if r["reached"])
        print(f"  phase {ph}: n={len(seg):>4}  到达目标={reach:>3}  撞墙率={bump:.2%}")

    print()
    print("语义记忆最活跃概念:", agent.semantic.salient(8))
    skill_state = {}
    for name, s in agent.pm.skills.items():
        skill_state[name] = {"成功率": round(s.mean, 2), "使用": s.uses}
    print("技能状态:", skill_state)
    wl = agent.engine.stats["world_loss"]
    print("元认知计算分配:", dict(agent.meta.compute_spent),
          " 世界模型损失:", None if wl is None else round(wl, 5))

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "trace.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in trace:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print()
    print(f"逐步认知痕迹已保存: {out_file}")


if __name__ == "__main__":
    main(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 600)
