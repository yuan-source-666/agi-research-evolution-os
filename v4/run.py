# -*- coding: utf-8 -*-
"""
run.py —— PRIMORDIA v4 · 启动入口
=====================================================================
用法：
  python run.py                          # 自主演化演示（headless 1500 拍）
  python run.py --mode live              # 交互模式：后台持续思考 + 人机对话
  python run.py --mode selftest          # 机械自检（均值聚合/安全门/纠正/锻造…）
  python run.py --mode demo --ticks 3000 --seed 11

运行要求：任意安装 Python ≥ 3.8 的普通电脑；零第三方依赖；无 GPU；
不依赖任何 LLM。全部状态与成长痕迹写入 out/ 目录。
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from primitives import (
    ActionContext, IrreversibleBlocked, Genome,
    make_primitive, INIT_ENERGY,
)
from evolution_engine import EvolutionEngine, FieldMedium, ResourceEnv, ToolForge
from communication_layer import CommunicationLayer, Reporter, write_dashboard


BANNER = r"""
 ██▓     ██░ ██  ▄▄▄       ██ ▄█▀▓█████  ██▀███   ███▄ ▄███▓
▓██▒    ▓██░ ██▒▒████▄     ██▄█▒ ▓█   ▀ ▓██ ▒ ██▒▓██▒▀█▀ ██▒
▒██░    ▒██▀▀██░▒██  ▀█▄  ▓███▄░ ▒███   ▓██ ░▄█ ▒▓██    ▓██░
▒██   ▒ ░▓█ ░██ ░██▄▄▄▄██ ▓██ █▄ ▒▓█  ▄ ▒██▀▀█▄  ▒██    ▒██
░ ████▓▒░░▓█▒░██▓ ▓█   ▓██▒▒██▒ █▄░▒████▒░██▓ ▒██▒▒██▒   ░██▒
"""


# ======================================================================
# 自检：对架构的关键断言做机械验证
# ======================================================================
def mode_selftest(args) -> int:
    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, True, ""))
        except Exception as exc:
            results.append((name, False, repr(exc)))

    eng = EvolutionEngine(seed=args.seed, out_dir=os.path.join(args.out, "selftest"))

    # 1. 无偏均值聚合：沉默者的 0 与喧哗者的 1 以同一分母进入平均
    def t_mean():
        m = FieldMedium()
        m.begin(4)
        m.submit([1.0] * 8)
        m.submit([1.0] * 8)
        m.submit([0.0] * 8)
        f = m.aggregate()
        assert all(abs(v - 0.5) < 1e-9 for v in f), f
    check("均值互作·无偏聚合（沉默者同分母）", t_mean)

    # 2. 原生安全门：不可逆动作缺人类令牌 ⇒ 物理上做不出
    def t_gate():
        eng._spawn("effect", tool="reset")   # 为检验安全门而诞生一名持有者
        eff = next(p for p in eng.population.values()
                   if p.kind == "effect" and p.genome.tool == "reset")
        ctx = ActionContext(env=eng.env, tick=1, context_sig="Po",
                            constraints=eng.constraints, confirm_token=None)
        try:
            eff.execute("reset", ctx)
            raise AssertionError("不可逆动作未被阻断")
        except IrreversibleBlocked:
            pass
    check("原生安全·不可逆动作三重确认门", t_gate)

    # 3. 约束册否决：人类纠正后同类动议丧失合格性
    def t_veto():
        eng.constraints.add("probe", "*", "测试纠正", eng.tick)
        eff = next(p for p in eng.population.values()
                   if p.kind == "effect" and p.genome.tool == "probe")
        ctx = ActionContext(env=eng.env, tick=2, context_sig="Po",
                            constraints=eng.constraints, confirm_token=None)
        r = eff.execute("probe", ctx)
        assert not r["ok"] and r["why"] == "constraint_veto", r
        eng.constraints.relax("probe", "*")
        r2 = eff.execute("probe", ctx)
        assert r2["ok"], r2
    check("原生安全·纠正约束使动作丧失合格性", t_veto)

    # 4. 纠正进入演化方向：问责窗参与者被扣能与敏化
    def t_correction():
        acc = eng.accountability
        acc.record(3, "nudge_plus", [], "", "Nb", 0.01)
        victims = [p for p in eng.population.values()][:5]
        acc.last["participants"] = [p.pid for p in victims]
        before = [(p.e, p.theta) for p in victims]
        eng.correct("nudge_plus", good=False, reason="太吵")
        after = [(p.e, p.theta) for p in victims]
        assert all(a[0] - b[0] >= 0.99 for a, b in zip(before, after)), (before, after)
        assert all(b[1] - a[1] >= 0.09 for a, b in zip(before, after))
    check("透明沟通·纠正纳入演化方向（问责惩罚）", t_correction)

    # 5. 工具锻造：反复受挫的动议 ⇒ 复合历史成功序列 ⇒ 新宏工具基元
    def t_forge():
        forge = ToolForge()
        for _ in range(6):
            forge.record_fail("nudge_minus", "Po")
        for _ in range(5):
            forge.record_success(["probe", "nudge_plus", "probe"])
        # 把世界摆到『正向轻推确实有用』的状态：x 远低于目标
        eng.env.x, eng.env.target = -0.5, 0.5
        made = forge.maybe_forge(eng)
        assert made and made["recipe"] == ["probe", "nudge_plus", "probe"], made
        child = eng.population.get(made["child"])
        assert child is not None and child.kind == "effect"
        assert child.genome.tool.startswith("macro:"), child.genome.tool
    check("工具锻造·受挫动议催生宏工具基元", t_forge)

    # 6. 生命周期：能量归零溶解；能量满线分裂
    def t_lifecycle():
        n0, d0 = len(eng.population), eng.deaths
        poor = next(iter(eng.population.values()))
        poor.e = -0.05   # 负值：连当拍红利也救不回来，账本判其溶解
        rich = next(p for p in eng.population.values() if p is not poor)
        rich.e = 16.0
        eng.step()
        assert eng.deaths == d0 + 1, (eng.deaths, d0)
        assert len(eng.population) >= n0, (len(eng.population), n0)
    check("生命周期·溶解与有丝分裂", t_lifecycle)

    # 7. 快照往返
    def t_snapshot():
        p = eng.save_snapshot(os.path.join(eng.out_dir, "snap_test.json"))
        eng2 = EvolutionEngine(seed=99, out_dir=os.path.join(eng.out_dir, "resume"))
        assert eng2.load_snapshot(p)
        assert len(eng2.population) == len(eng.population)
    check("持续记录·快照保存与恢复", t_snapshot)

    # 8. 短程自主性：50 拍内出现自发事件
    def t_emerge():
        e2 = EvolutionEngine(seed=args.seed + 1,
                             out_dir=os.path.join(args.out, "emerge"))
        for _ in range(60):
            e2.step()
        ev = open(e2.p_events, encoding="utf-8").read()
        assert "BOND_FORMED" in ev or "MITOSIS" in ev, "60拍内无自发结构事件"
    check("自驱动演化·无外部指令下自发产生结构事件", t_emerge)

    # 9. 自我修正：监察惊讶异常 ⇒ 阻尼最惊讶者并留档
    def t_selfcorr():
        e3 = EvolutionEngine(seed=args.seed + 2,
                             out_dir=os.path.join(args.out, "selfcorr"))
        e3.tick = 400
        metas = [p for p in e3.population.values() if p.kind == "meta"]
        for m in metas:
            m.surprise_ema = 1.2
        before = {m.pid: (m.genome.gain, m.theta) for m in metas}
        e3.err_baseline = 0.1
        e3._self_correct()
        changed = any(abs(m.genome.gain - before[m.pid][0]) > 1e-6 for m in metas)
        ev = open(e3.p_events, encoding="utf-8").read()
        assert changed and "SELF_CORRECTION" in ev, (changed, "SELF_CORRECTION" in ev)
    check("自我修正·监察异常触发阻尼并留档", t_selfcorr)

    print("=" * 62)
    ok = 0
    for name, passed, err in results:
        mark = "PASS" if passed else "FAIL"
        ok += passed
        print("[%s] %s%s" % (mark, name, ("　→ " + err) if err else ""))
    print("-" * 62)
    print("自检结果：%d/%d 通过" % (ok, len(results)))
    return 0 if ok == len(results) else 1


# ======================================================================
# 演示：headless 自主演化
# ======================================================================
def mode_demo(args) -> int:
    eng = EvolutionEngine(seed=args.seed, out_dir=args.out)
    if args.resume and eng.load_snapshot(args.resume):
        print("已从快照恢复：tick %d" % eng.tick)
    rep = Reporter(eng)
    print(BANNER)
    print("PRIMORDIA v4 —— 均值场上的基元社会开始演化（seed=%d，%d 拍）"
          % (args.seed, args.ticks))
    print("初始种群 %d 基元。观察者不在场；以下一切由内在动力学自发发生。" 
          % len(eng.population))
    print("-" * 62)
    t0 = time.time()
    for i in range(1, args.ticks + 1):
        eng.step()
        if i % 100 == 0:
            f = eng.status_facts()
            print("tick %5d｜pop %3d｜键 %4d｜胜任度 %.3f｜唤醒 %.3f｜"
                  "生 %d 亡 %d｜行动 %s"
                  % (f["tick"], f["pop"], f["bonds"], f["competence"],
                     f["arousal"], f["births"], f["deaths"],
                     sum(f["actions"].values()) if f["actions"] else 0))
        if i % 500 == 0:
            write_dashboard(eng)
    dt = time.time() - t0
    dash = write_dashboard(eng)
    snap = eng.save_snapshot()
    print("-" * 62)
    print(rep.status_report())
    print()
    print(rep.growth_summary())
    print()
    print(rep.tools_report())
    print()
    print("用时 %.2f 秒（%.0f 拍/秒）。成长档案：" % (dt, args.ticks / max(dt, 1e-9)))
    for p in (os.path.join(args.out, "growth_log.jsonl"),
              os.path.join(args.out, "events.jsonl"), dash, snap):
        print("  " + p)
    return 0


# ======================================================================
# 交互模式：后台持续思考 + 人机对话
# ======================================================================
def mode_live(args) -> int:
    eng = EvolutionEngine(seed=args.seed, out_dir=args.out)
    if args.resume and eng.load_snapshot(args.resume):
        print("已从快照恢复：tick %d" % eng.tick)
    layer = CommunicationLayer(eng)
    layer.interval = args.interval
    lock = threading.Lock()
    stop = threading.Event()

    def thinker():
        last_dash = 0
        last_save = eng.tick
        while not stop.is_set():
            if layer.paused:
                time.sleep(0.05)
                continue
            with lock:
                eng.step()
                if eng.tick - last_dash >= 200:
                    last_dash = eng.tick
                    write_dashboard(eng)
                if eng.tick - last_save >= 1200:
                    last_save = eng.tick
                    eng.save_snapshot()
            time.sleep(layer.interval)

    th = threading.Thread(target=thinker, daemon=True)
    th.start()
    print(BANNER)
    print("PRIMORDIA v4 交互模式。基元社会正在你身后持续思考"
          "（每拍约 %.2f 秒）。" % layer.interval)
    print("对我说『帮助』可看全部指令；『报告』要体检；『为什么』听解释；")
    print("『纠正：<动作>』会真实改变我的演化方向。Ctrl+C 或『退出』离开。")
    print("-" * 62)
    try:
        while True:
            try:
                text = input("观察者> ").strip()
            except EOFError:
                print()
                break
            if not text:
                continue
            with lock:
                resp = layer.handle(text)
            print(resp)
            if layer.quit_flag:
                break
    except KeyboardInterrupt:
        print()
        layer.quit_flag = True
    finally:
        stop.set()
        th.join(timeout=3)
        with lock:
            dash = write_dashboard(eng)
            snap = eng.save_snapshot()
            layer._tlog("system", "会话结束，快照已保存。")
        print("已存档：%s" % snap)
        print("仪表盘：%s" % dash)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="PRIMORDIA v4 —— 结构即智能的基元社会")
    ap.add_argument("--mode", choices=["demo", "live", "selftest"], default="demo")
    ap.add_argument("--ticks", type=int, default=1500, help="demo 模式演化拍数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="输出目录（默认 v4/out）")
    ap.add_argument("--interval", type=float, default=0.25, help="live 模式每拍秒数")
    ap.add_argument("--resume", default=None, help="从此快照恢复")
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    if args.out is None:
        args.out = os.path.join(base, "out")
    os.makedirs(args.out, exist_ok=True)
    if args.mode == "selftest":
        return mode_selftest(args)
    if args.mode == "live":
        return mode_live(args)
    return mode_demo(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                input(chr(10) + "[发生错误] 已打印堆栈如上；按回车键退出...")
        except Exception:
            pass
        raise SystemExit(1)
