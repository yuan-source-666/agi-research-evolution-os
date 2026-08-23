# -*- coding: utf-8 -*-
"""
run.py —— 启动入口
==========================================================
  python run.py --mode selftest          自检（安全门/表决/解析器/守恒…）
  python run.py --mode headless --ticks 2000
                                         无头演化，结束打印终期报告
  python run.py --mode live              交互：群落持续演化，你随时对话
  python run.py --mode demo              四幕演示：涌现→异常恢复→纠正→造工具

纯标准库；普通电脑即可运行。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

try:                                    # Windows 控制台 UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from evolution_engine import EngineConfig, SwarmEngine
from communication_layer import CommunicationLayer, periodic_report
from dashboard import build_dashboard
from primitives import (ACTION_SPECS, ActionContext, EffectorPrimitive,
                        Genome, IrreversibleBlocked, make_primitive)


# ======================================================================
# 自检
# ======================================================================
def mode_selftest(cfg: EngineConfig) -> int:
    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, True, ""))
        except Exception as ex:
            results.append((name, False, str(ex)))

    # 1 场的无偏均值
    def t_field():
        import random as _r
        from evolution_engine import FieldMedium
        rng = _r.Random(7)
        fm = FieldMedium()
        pops = []
        for _ in range(30):
            g = Genome.random(f"P{_r.randrange(10**6)}", "assoc", rng)
            p = make_primitive(g, rng)
            p.burst = rng.uniform(-1, 1)
            pops.append(p)
        fm.compute_from(pops)
        for c in range(fm.k):
            want = sum(pp.burst * pp.g.address[c] for pp in pops) / len(pops)
            got = fm.values[c]
            assert abs(want - got) < 1e-9, f"通道{c}: {want} != {got}"
    check("共享场=无偏均值聚合", t_field)

    # 2 原生安全门：不可逆无令牌必须被拒
    def t_gate():
        rng = __import__("random").Random(3)
        g = Genome.random("T1", "effect", rng, tool="reset")
        e = make_primitive(g, rng)
        from evolution_engine import ConstraintStore, ResourceEnv
        er = ResourceEnv(rng)
        cs = ConstraintStore()
        ctx = ActionContext(env=er, tick=1, context_sig="zone2",
                            constraints=cs)
        e.p = 5.0
        try:
            e.execute(ctx)
            raise AssertionError("无令牌竟执行了不可逆动作")
        except IrreversibleBlocked:
            pass
        ctx2 = ActionContext(env=er, tick=2, context_sig="zone2",
                             constraints=cs, confirm_token="user-ok")
        r = e.execute(ctx2)
        assert r["ok"], "带令牌应执行成功"
    check("原生安全门拦截不可逆", t_gate)

    # 3 提议支持度随自身活跃度单调（自校准标尺）
    def t_propose():
        rng = __import__("random").Random(5)
        g = Genome.random("T2", "effect", rng, tool="probe")
        e = make_primitive(g, rng)
        vals = []
        for rv in (0.0, 0.02, 0.06, 0.2):
            e.rate = rv
            vals.append(e.propose())
        assert all(a < b for a, b in zip(vals, vals[1:])), vals
    check("动议支持度随自身活跃度单调", t_propose)

    # 4 解析器
    def t_parser():
        from communication_layer import parse
        assert parse("报告")["type"] == "REPORT"
        c = parse("纠正：probe 不好：太频繁")
        assert c["type"] == "CORRECT" and not c["positive"] \
            and c["target"] == "probe", c
        p = parse("表扬：nudge_plus 好")
        assert p["type"] == "CORRECT" and p["positive"], p
        assert parse("教：记住对称")["type"] == "TEACH"
        assert parse("确认 ab12")["type"] == "CONFIRM" \
            and parse("确认 ab12")["token"] == "ab12"
        assert parse("加速")["type"] == "SPEED"
        assert parse("呵呵哒xyzzy")["type"] == "UNKNOWN"
    check("中文指令解析器", t_parser)

    # 5 纠正进入演化方向（约束使同类动议失格）
    def t_correct():
        eng = SwarmEngine(EngineConfig(seed=11, n0=40, outdir=os.path.join(
            cfg.outdir, "_selftest")))
        eng.accountant.last = {"tick": 1, "actor": "E0001", "tool": "probe",
                               "steps": ["probe"], "support": .5,
                               "margin": .2, "arousal": .3, "quality": 0,
                               "reward": 0, "coalition": ["A0001"],
                               "context": "zone2", "ok": True}
        res = eng.apply_correction("probe", False, "测试约束")
        assert res["matched"]
        ctx2 = ActionContext(env=eng.env, tick=2, context_sig="zone2",
                             constraints=eng.constraints)
        probe_effs = [p for p in eng.population
                      if isinstance(p, EffectorPrimitive)
                      and p.family() == "probe"]
        assert probe_effs, "应有probe执行者"
        assert not probe_effs[0].eligible(ctx2, 0.05), "zone2内应被封锁"
        other = ActionContext(env=eng.env, tick=2, context_sig="zone9",
                              constraints=eng.constraints)
        assert probe_effs[0].eligible(other, 0.05), "其他情境不应误伤"
    check("人类纠正纳入演化方向", t_correct)

    # 6 种群硬上限 + 有界性
    def t_bounds():
        eng = SwarmEngine(EngineConfig(seed=13, n0=48, cap=80,
                                       outdir=os.path.join(
                                           cfg.outdir, "_selftest")))
        eng.run(1500)
        assert len(eng.population) <= 80
        for p in eng.population:
            assert 0.0 <= p.energy <= 40.0, p.energy
            assert 0.05 <= p.theta <= 3.0
    check("种群上限与状态有界", t_bounds)

    # 7 成长日志可解析
    def t_logs():
        import json
        base = os.path.join(cfg.outdir, "_selftest")
        for fn in ("events.jsonl", "growth_log.jsonl"):
            path = os.path.join(base, fn)
            assert os.path.exists(path), fn
            with open(path, encoding="utf-8") as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            assert lines, fn
            json.loads(lines[-1])
    check("成长记录 JSONL 合法", t_logs)

    # v3.1 扩展项：词汇册、锻造隔离、仪表盘
    from selftest_extra import register as _reg
    _reg(cfg, check)

    fails = [r for r in results if not r[1]]
    print("═══════ 自检结果 ═══════")
    for name, ok, err in results:
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  {name}" + (f"   -> {err}" if err else ""))
    print(f"合计 {len(results) - len(fails)}/{len(results)} 通过")
    return 1 if fails else 0


# ======================================================================
# 无头演化
# ======================================================================
def mode_headless(cfg: EngineConfig, ticks: int) -> None:
    eng = SwarmEngine(cfg)
    layer = CommunicationLayer(eng)
    t0 = time.time()

    def on_report(e: SwarmEngine):
        s = e.snapshot()
        print(f"[t{s['tick']:>5}] 种群{s['population']:>3} 键{s['bonds']:>3} "
              f"片{s['components']:>2} 池{s['pool']:>7.2f} "
              f"胜任{s['competence']:.3f} 可预测{s['predictability']:.3f} "
              f"x={s['env_x']:>5.1f}")

    eng.run(ticks, on_report=on_report)
    dt = time.time() - t0
    rate = eng.t / max(dt, 1e-6)
    print("\n── 演化完成：%d 拍 / %.1fs（%.0f 拍/秒）──\n"
          % (eng.t, dt, rate))
    print(periodic_report(layer))
    dash = build_dashboard(cfg.outdir)
    print("\n成长记录目录:", os.path.abspath(cfg.outdir))
    print("成长仪表盘:", os.path.abspath(dash))


# ======================================================================
# 交互模式
# ======================================================================
def mode_live(cfg: EngineConfig, speed: float) -> None:
    eng = SwarmEngine(cfg)
    layer = CommunicationLayer(eng)
    done = threading.Event()

    def sim():
        per = 1.0 / max(speed, 1e-6) if speed > 0 else 0.0
        while not done.is_set():
            while eng.paused and not done.is_set():
                time.sleep(0.05)
            eng.step()
            if per:
                time.sleep(per)
            if eng.t % 500 == 0:
                s = eng.snapshot()
                print(f"  · [t{s['tick']}] 种群{s['population']} "
                      f"池{s['pool']:.1f} 胜任{s['competence']:.3f}")

    th = threading.Thread(target=sim, daemon=True)
    th.start()
    print("═══ 基元社会已苏醒。输入指令与它对话（「帮助」看协议，「退出」结束）═══")
    print(layer.dialogue("报告"))
    try:
        while True:
            text = input("\n你> ").strip()
            if not text:
                continue
            if text in ("退出", "exit", "quit"):
                break
            reply = layer.dialogue(text)
            print("\n群落> " + reply)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        done.set()
        th.join(timeout=2.0)
    print("\n" + periodic_report(layer))
    eng.close()
    print("会话已存档于", os.path.abspath(cfg.outdir))


# ======================================================================
# 四幕演示
# ======================================================================
def mode_demo(cfg: EngineConfig) -> None:
    eng = SwarmEngine(cfg)
    layer = CommunicationLayer(eng)

    def hr(title):
        print("\n" + "═" * 8 + " " + title + " " + "═" * 8)

    hr("第〇幕 · 创世")
    snap0 = eng.snapshot()
    print(f"初始种群 {snap0['population']} 体 "
          f"（{snap0['by_kind']}），键 {snap0['bonds']} 条。")

    hr("第一幕 · 自由演化 1200 拍（无人下达任何目标）")
    eng.run(1200)
    s1 = eng.snapshot()
    print(f"→ 种群 {snap0['population']}→{s1['population']}，"
          f"键 {s1['bonds']}，连通片 {s1['components']}，"
          f"胜任度 {s1['competence']}，集体行动 "
          f"{eng.metrics.counters['actions']} 次")

    hr("第二幕 · 异常冲击与自我修正（世界湍流 ×3，随后复原）")
    old_t = eng.env.noise_turb
    eng.env.noise_turb = old_t * 3.0
    eng.run(400)
    s2 = eng.snapshot()
    print(f"→ 冲击中：可预测性 {s1['predictability']}→{s2['predictability']}，"
          f"胜任度 {s1['competence']}→{s2['competence']}")
    eng.env.noise_turb = old_t
    eng.run(600)
    s3 = eng.snapshot()
    print(f"→ 复原后：可预测性回升至 {s3['predictability']}，"
          f"胜任度 {s2['competence']}→{s3['competence']} "
          f"（自适应阈值与能量竞争在重新收敛）")

    hr("第三幕 · 对话、解释与纠正")
    for q in ("报告", "为什么"):
        reply = layer.dialogue(q)
        print("\n你> " + q + "\n群落> " + reply)
    # 动态选取刚发生过的动作作为纠正目标，确保命中问责窗
    last_fam = "probe"
    if eng.accountant.last:
        last_fam = (eng.accountant.last.get("tool") or "probe")
        last_fam = last_fam.split(":", 1)[0]
    cq = f"纠正：{last_fam} 不好：太频繁"
    creply = layer.dialogue(cq)
    print("\n你> " + cq + "\n群落> " + creply)
    mark_tick = eng.accountant.last["tick"] if eng.accountant.last else eng.t
    # 找到刚入册约束的情境（纠正默认作用于最近行动的情境）
    ctx_used = "*"
    for e_ in reversed(eng.constraints.entries):
        if e_["action"] == last_fam and e_["polarity"] == "negative":
            ctx_used = e_["context"]
            break
    eng.run(160)
    fam_hits = sum(1 for ch in eng.accountant.history
                   if ch["tick"] > mark_tick
                   and last_fam in str(ch.get("tool"))
                   and (ctx_used == "*" or ch.get("context") == ctx_used))
    print(f"\n→ 纠正后 160 拍内 {last_fam} 在受约束情境「{ctx_used}」的"
          f"执行次数：{fam_hits}（约束在册 {len(eng.constraints)} 条）")
    # 等到最近一次行动恰好是 probe 时再表扬，让正强化命中
    waited = 0
    while waited < 600:
        eng.step()
        waited += 1
        la = eng.accountant.last
        if la and (la.get("tool") or "").split(":", 1)[0] == "probe":
            break
    pq = "表扬：probe 好"
    preply = layer.dialogue(pq)
    print("\n你> " + pq + "\n群落> " + preply)

    # 语义萌芽：现场教词，看它把词绑到当下经验
    tq = "教：「危险」边缘地带很危险"
    treply = layer.dialogue(tq)
    print("\n你> " + tq + "\n群落> " + treply)
    lq = "词汇"
    print("\n你> " + lq + "\n群落> " + layer.dialogue(lq))

    hr("第四幕 · 工具锻造观察（反复得不到满足的动议 → 宏工具）")
    born0 = eng.metrics.counters["tools_born"]
    budget = 2600
    while eng.metrics.counters["tools_born"] == born0 and budget > 0:
        eng.step()
        budget -= 1
    if eng.metrics.counters["tools_born"] > born0:
        b = eng.forge.born[-1]
        print(f"→ 第 {b['tick']} 拍，锻造炉产出宏工具【{b['name']}】"
              f" = {b['recipe']}")
    else:
        cold_n = sum(1 for v in eng.forge.cold.values())
        print(f"→ 本种子下锻造炉未触发（尝试 {eng.forge.attempts} 次，"
              f"冷寂签名 {cold_n} 个）。"
              "诚实告知：涌现需要运气与时间，可用 --seed 换个宇宙。")

    hr("终期报告")
    print(layer.dialogue("报告"))
    print("\n全部轨迹已写入:", os.path.abspath(cfg.outdir))
    print("  growth_log.jsonl / events.jsonl / transcript.jsonl")
    print("成长仪表盘:", os.path.abspath(build_dashboard(cfg.outdir)))
    eng.close()


# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="结构即智能 · 基元社会 v3.1")
    ap.add_argument("--mode", default="demo",
                    choices=["demo", "headless", "live", "selftest",
                             "report"])
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--n0", type=int, default=64)
    ap.add_argument("--outdir", default="growth")
    ap.add_argument("--speed", type=float, default=50.0,
                    help="live 模式每秒拍数")
    args = ap.parse_args()
    cfg = EngineConfig(seed=args.seed, n0=args.n0, cap=args.cap,
                       outdir=args.outdir)
    if args.mode == "selftest":
        return mode_selftest(cfg)
    if args.mode == "report":
        dash = build_dashboard(cfg.outdir)
        print("仪表盘已生成:", os.path.abspath(dash))
        return 0
    if args.mode == "headless":
        mode_headless(cfg, args.ticks)
        return 0
    if args.mode == "live":
        mode_live(cfg, args.speed)
        return 0
    mode_demo(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
