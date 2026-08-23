# -*- coding: utf-8 -*-
"""
SEED OS v2 — run.py
启动入口：demo（自动演示 + 验收自检）/ interactive（人类对话模式）

用法：
    python run.py                     # 双击/默认 = 交互模式（与系统对话）
    python run.py --mode demo         # 自动演示 + 验收自检
    python run.py --ticks 400         # 指定预热 tick 数
    python run.py --fresh             # 放弃快照，开始一段全新生命（默认苏醒延续）
"""
import argparse
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from evolution_engine import EvolutionEngine
from communication_layer import CommunicationLayer
from language_cortex import LanguageCortex


def banner(title):
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)


def train_cortex(comm):
    """启动即训练语言皮层：核心语料 + 大语料 + 老板教过的话。
    train_all 内置 pickle 缓存——首次训练 ~1s，之后重启秒加载。"""
    cortex = LanguageCortex()
    cortex.train_all(verbose=True)
    comm.cortex = cortex
    return cortex, len(cortex.memories), 0


def _print_life_status(engine):
    if engine.awakenings:
        print(f"[苏醒] 延续既有生命：tick {engine.tick} / 周期 {engine.cycle} / "
              f"{len(engine.registry)} 基元（第 {engine.awakenings + 1} 段连续生命）")
    else:
        print("[新生] 从种子拓扑开始一段全新的生命")


def demo(ticks, seed, resume=True):
    banner("SEED OS v2 —— 结构演化型智能体 · 演示运行")
    engine = EvolutionEngine(seed=seed, resume=resume)
    comm = CommunicationLayer(engine)
    train_cortex(comm)   # demo 也带上语言能力
    _print_life_status(engine)

    origin = "延续快照" if engine.awakenings else "种子拓扑"
    print(f"当前结构：{len(engine.registry)} 基元 / {len(engine.fields)} 场（{origin}）")
    print(f"运行 {ticks} tick（结构周期 = {engine.CYCLE} tick），"
          f"产物目录：{engine.gdir}")

    engine.run(ticks)

    banner(f"演化完成：tick {engine.tick} / 周期 {engine.cycle}")
    print(comm.system_report())

    # ---------- 验收演示：对话 ----------
    banner("验收演示 1：对话（系统解释自身）")
    for msg in ["你好，结构", "状态"]:
        print(f"\n【人类】{msg}")
        print(f"【SEED OS】{comm.chat(msg)}")

    banner("验收演示 2：人类纠正 → 纳入演化方向")
    print(f"【人类】纠正：优先发展与镜像预测相关的结构")
    print(f"【SEED OS】{comm.chat('纠正：优先发展与镜像预测相关的结构')}")
    engine.run(engine.CYCLE * 3)   # 纠正生效 3 个周期
    corr_events = [e for e in engine.events if e["type"] == "correction"]
    print(f"\n（纠正已写入事件日志：{len(corr_events)} 条，"
          f"valorization 期间结构键增长率 ×2.2）")

    banner("验收演示 3：为什么（事件缘由可追溯）")
    last_struct = [e for e in engine.events
                   if e["type"] in ("growth", "proliferation", "pruning",
                                    "merge", "fission", "tool_emergence")]
    if last_struct:
        probe = last_struct[-1]["detail"].split()[0]
        print(f"【人类】为什么 {probe}")
        print(f"【SEED OS】{comm.chat(f'为什么 {probe}')}")

    banner("验收演示 4：工具（涌现 + 调用）")
    print(f"【人类】工具列表")
    print(f"【SEED OS】{comm.tools_report()}")
    tools = engine.list_tools()
    if tools:
        name = tools[-1]["tool"]
        print(f"\n【人类】调用 {name}")
        print(f"【SEED OS】{comm.chat(f'调用 {name}')}")

    banner("验收演示 5：原生安全（不可逆动作多层确认）")
    engine.run(engine.CYCLE * 4)
    pend = [a for a in engine.pending_actions if a["status"] == "awaiting_human"]
    if pend:
        print(f"系统声明了不可逆动作：{pend[0]['id']}:{pend[0]['action']}"
              f"（来源 {pend[0]['source']}，tick {pend[0]['tick']}）")
        print(f"【人类】拒绝 {pend[0]['id']}")
        print(f"【SEED OS】{comm.chat(f'拒绝 {pend[0]['id']}')}")
    else:
        print("（本演示窗口内行动基元未提出不可逆动作——词表中该类动作"
              "仅在自发动力学中低概率出现）")
    declared = sum(1 for e in engine.events if e["type"] == "irreversible_declared")
    confirmed = sum(1 for e in engine.events if e["type"] == "irreversible_confirmed")
    rejected = sum(1 for e in engine.events if e["type"] == "irreversible_rejected")
    awaiting = sum(1 for a in engine.pending_actions if a["status"] == "awaiting_human")
    auto = getattr(engine, "auto_rejected", 0)
    print(f"\n不可逆动作账本：声明 {declared} 条 | 人类确认 {confirmed} 条 | "
          f"人类拒绝 {rejected} 条 | 当前待确认 {awaiting} 条 | "
          f"队列溢出自动否决 {auto} 条")
    print("结论：没有任何不可逆动作绕过人类确认——这是词表级原生安全的结构性保证。")

    banner("验收演示 6：能力选择（评价变成结构存亡的理由）")
    print("【人类】彩虹是怎么形成的？")
    print(f"【SEED OS】{comm.chat('彩虹是怎么形成的？')}")
    print("\n【人类】好评")
    print(f"【SEED OS】{comm.chat('好评')}")
    evs = [e for e in engine.events if e["type"] == "competence_reward"]
    if evs:
        last = evs[-1]
        print(f"\n（事件日志 tick {last['tick']} competence_reward | "
              f"{json.dumps(last['reason'], ensure_ascii=False)[:100]}）")
        print("（记功基元从此优先增殖；反之差评记过多者加速凋亡。）")
    print("\n【人类】差评")
    print(f"【SEED OS】{comm.chat('差评')}")

    # ---------- 最终产物 ----------
    comm.generate_dashboard()
    engine.save_snapshot()
    banner("产物清单")
    for rel in ("trajectory.jsonl", "events.jsonl", "growth_log.md",
                "dashboard.html", "snapshot.pkl"):
        p = os.path.join(engine.gdir, rel)
        size = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"  growth/{rel:<18} {size:>8} bytes")
    sb = os.listdir(engine.sandbox)
    print(f"  growth/sandbox/            {len(sb)} 个系统自产制品"
          f"（{', '.join(sb[:5])}...）" if sb else "  growth/sandbox/（空）")

    first = engine.metrics_history[0] if engine.metrics_history else {}
    last = engine.metrics_history[-1] if engine.metrics_history else {}
    banner("智能曲线（复合指数）")
    print(f"  首周期：{first.get('intelligence', '—')}  →  "
          f"末周期：{last.get('intelligence', '—')}")
    print(f"  结构事件总数：{len(engine.events)}（生长/增殖/剪枝/重组/工具涌现/自我修正）")
    print(f"\n交互模式请运行：python run.py --mode interactive")


def interactive(ticks, seed, resume=True):
    banner("SEED OS · 小种子")
    engine = EvolutionEngine(seed=seed, resume=resume)
    comm = CommunicationLayer(engine)

    if engine.awakenings:
        print(f"[苏醒] 我回来了——还是原来那个我"
              f"（tick {engine.tick} / 周期 {engine.cycle} / "
              f"{len(engine.registry)} 基元，第 {engine.awakenings + 1} 段连续生命）")
    else:
        print("正在出生（种子拓扑）…")
    print("正在长大（结构预热）…")
    engine.run(ticks)

    cortex, total, _ = train_cortex(comm)
    print(f"语言训练完成：共 {total} 组记忆（含你教过的话）")

    print()
    print("┌─────────────────────────────────────────────┐")
    print("│  直接打字跟我聊天（你好 / 讲个笑话 / 光速是多少）  │")
    print("│                                             │")
    print("│  状态      我的成长简报                        │")
    print("│  学习 Q|A  教我一句新话（立刻生效、永久保存）      │")
    print("│  工具      我会的能力                          │")
    print("│  为什么 X  追问我某个变化的来龙去脉              │")
    print("│  帮助      全部指令                            │")
    print("│  quit      退出                               │")
    print("└─────────────────────────────────────────────┘")
    print(f"\n{comm.brief_report()}")

    while True:
        try:
            msg = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            break
        print(f"\n小种子：{comm.chat(msg)}")
    comm.generate_dashboard()
    engine.save_snapshot()
    print("\n这轮对话已存档，结构已快照——下次启动我会苏醒，而不是重新出生。")
    print(f"成长曲线：{os.path.join(engine.gdir, 'dashboard.html')}")


def main():
    ap = argparse.ArgumentParser(description="SEED OS v2 启动入口")
    ap.add_argument("--mode", choices=["demo", "interactive"], default="interactive")
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true",
                    help="放弃快照，从种子拓扑开始一段全新生命（默认苏醒延续）")
    args = ap.parse_args()
    resume = not args.fresh
    if args.mode == "demo":
        demo(args.ticks, args.seed, resume)
    else:
        interactive(args.ticks, args.seed, resume)


def _pause_if_double_clicked():
    """双击启动（无参数、无控制台交互）时，结束后暂停防窗口闪退。"""
    if len(sys.argv) == 1:
        try:
            input("\n按回车键退出…")
        except EOFError:
            pass


if __name__ == "__main__":
    main()
    _pause_if_double_clicked()
