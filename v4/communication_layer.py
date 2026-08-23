# -*- coding: utf-8 -*-
"""
communication_layer.py —— PRIMORDIA v4 · 通信与自述模块
=====================================================================
语言不是基元的天赋；Reporter 是架在统计事实上的一块单向玻璃。
一切对外语句均由 MetricsRegistry / 事件流的量化事实即时渲染——
每一个数字都可对照 out/growth_log.jsonl 与 out/events.jsonl 验证，
拒绝空泛表述。

协议文法（人类 → 系统）：
  报告|状态|体检      → 全量量化状态报告
  为什么|解释         → 最近一次集体行动的完整因果链
  工具                → 已注册工具 + 宏配方 + 锻造统计
  成长|演化史|轨迹     → 智能曲线对比 + 结构时间线里程碑
  你是谁|自我描述      → 种群自述（含最优秀基元的自我刻画）
  教：<文本>          → 教导注入感知层
  纠正：<动作> [理由]  → 负校正进入约束册与问责窗
  表扬：<动作> [理由]  → 正校正
  加速/减速 [n]       → 演化节奏（live 模式）
  暂停 / 继续         → 冻结 / 恢复动力学
  看板                → ANSI 实时仪表盘快照
  保存                → 写状态快照
  帮助                → 本协议全文
解析失败时诚实告知可用指令集，绝不编造。
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import deque
from typing import Dict, List, Optional

from evolution_engine import EvolutionEngine

BLOCKS = "▁▂▃▄▅▆▇█"


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------
def sparkline(values: List[float], width: int = 40) -> str:
    vals = [v for v in values[-width:]]
    if not vals:
        return "（暂无数据）"
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return BLOCKS[3] * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / (hi - lo) * (len(BLOCKS) - 1) + 0.5)
        out.append(BLOCKS[max(0, min(len(BLOCKS) - 1, idx))])
    return "".join(out)


def svg_sparkline(values: List[float], w: int = 560, h: int = 90,
                  color: str = "#4f8", label: str = "") -> str:
    if not values:
        return "<p class='muted'>暂无数据</p>"
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    pts = []
    n = len(values)
    for i, v in enumerate(values):
        x = 4 + (w - 8) * (i / max(1, n - 1))
        y = h - 6 - (h - 16) * ((v - lo) / rng)
        pts.append("%.1f,%.1f" % (x, y))
    head = ("%.4g" % hi) + "　最低 " + ("%.4g" % lo)
    return ("<svg width='%d' height='%d' class='chart'>"
            "<polyline fill='none' stroke='%s' stroke-width='2' points='%s'/>"
            "<text x='6' y='14' fill='%s' font-size='11'>%s · %s</text></svg>"
            % (w, h, color, " ".join(pts), color, label, head))


def _tail_events(path: str, etype: Optional[str] = None,
                 n: int = 5) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    out: List[Dict] = []
    for line in reversed(lines):
        if len(out) >= n:
            break
        try:
            row = json.loads(line)
        except Exception:
            continue
        if etype is None or row.get("type") == etype:
            out.append(row)
    return out


def _all_events(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]
    except Exception:
        return []


# ======================================================================
# Reporter：把统计事实渲染为人类可读的报告
# ======================================================================
class Reporter:
    def __init__(self, engine: EvolutionEngine):
        self.engine = engine

    # ---- 状态报告 ----
    def status_report(self) -> str:
        e = self.engine
        f = e.status_facts()
        nl = chr(10)
        env = f["env"]
        acts = f["actions"] or {}
        act_s = ("，".join("%s×%d" % (k, v) for k, v in sorted(acts.items()))
                 if acts else "尚无")
        lines = [
            "════ PRIMORDIA v4 · 状态报告（tick %d）═══" % f["tick"],
            "种群：%d 基元（sensor %d｜assoc %d｜memory %d｜meta %d｜effect %d）"
            % (f["pop"], f["by_kind"]["sensor"], f["by_kind"]["assoc"],
               f["by_kind"]["memory"], f["by_kind"]["meta"], f["by_kind"]["effect"]),
            "结构：%d 条键｜%d 个连通片｜最大片占比 %.2f｜平均度 %.2f"
            % (f["bonds"], f["components"], f["largest_frac"], f["mean_degree"]),
            "场强（均值聚合后 8 通道）：%s" % (fld_str(f["field"]),),
            "能量经济：本拍环境收益 %.4f，人均红利 %.5f｜生 %d / 亡 %d"
            % (f["income"], f["dividend_last"], f["births"], f["deaths"]),
            "智能曲线：胜任度 %.3f｜新颖度(近100拍变异) %d｜唤醒度 %.3f"
            % (f["competence"], f["novelty"], f["arousal"]),
            "世界：x=%.3f 目标=%.3f %s｜稳定度 %.2f"
            % (env["x"], env["target"], "在带内" if env["in_band"] else "带外",
               env["stability"]),
            "集体行动累计：%s｜记忆重放 %d 次" % (act_s, f["replays"]),
            "原生安全：生效约束 %d 条｜锻造宏工具 %d 件"
            % (f["constraints_active"], e.forge.forged),
            "—— 以上数字均可在 out/growth_log.jsonl 与 out/events.jsonl 中核对。",
        ]
        return nl.join(lines)

    # ---- 解释最近行动 ----
    def explain_last(self) -> str:
        e = self.engine
        nl = chr(10)
        acc = e.accountability.last
        parts: List[str] = ["════ 最近一次集体行动的因果链 ═══"]
        acted = _tail_events(e.p_events, "ACTION_EXECUTED", 1)
        blocked = _tail_events(e.p_events, "IRREVERSIBLE_BLOCKED", 1)
        vetoed = _tail_events(e.p_events, "MOTION_VETOED", 1)
        if acted:
            a = acted[0]
            parts += [
                "触发：tick %d 集体唤醒度/周期到达表决条件" % a["tick"],
                "胜出动议：%s（支持度边际 %.3f）——由执行基元 [%s] 以自身势能 tanh(p) 提出"
                % (a["family"], a["margin"], a["winner"]),
                "法定过程：%d 个合格动议参与投票，并列时公平抽签（无偏）；情境签名 %s"
                % (a["participants"], a["sig"]),
                "执行序列：" + " → ".join(a.get("steps", [])),
                "结果：目标贴近度变化 %+0.4f；奖励按联盟（%d 人）人均均分 %+.5f"
                % (a["prox_delta"], a["alliance"], a["reward_each"]),
                "问责窗已记录全部 %d 名参与者，供后续纠正追溯。"
                % a["participants"],
            ]
        else:
            parts.append("尚未有过通过的动议。")
        if blocked:
            parts.append("安全事件：tick %d 曾提出不可逆动作被原生安全门拒绝（%s）"
                         % (blocked[0]["tick"], blocked[0].get("why", "")[:60]))
        if vetoed:
            parts.append("约束生效：tick %d 动议 %s 因人类纠正记录（情境 %s）丧失合格性"
                         % (vetoed[0]["tick"], vetoed[0].get("family"),
                            vetoed[0].get("sig")))
        return nl.join(parts)

    # ---- 工具清单 ----
    def tools_report(self) -> str:
        e = self.engine
        from primitives import ACTION_SPECS
        nl = chr(10)
        lines = ["════ 工具清单 ═══",
                 "内置工具（可逆性是定义的一部分）："]
        for name, spec in ACTION_SPECS.items():
            tag = "可逆" if spec["reversible"] else "不可逆·需三重确认"
            lines.append("  %-12s %s｜%s" % (name, tag, spec["desc"]))
        holders: Dict[str, int] = {}
        for p in e.population.values():
            if p.kind == "effect" and p.genome.tool:
                holders[p.genome.tool] = holders.get(p.genome.tool, 0) + 1
        lines.append("当前持有者：" + ("，".join("%s×%d" % kv for kv in sorted(holders.items()))
                                  if holders else "无"))
        macros = {k: v for k, v in e.forge.recipes.items()}
        if macros:
            lines.append("锻造宏工具：")
            for k, v in macros.items():
                lines.append("  %s = %s" % (k, " → ".join(v)))
        else:
            lines.append("锻造宏工具：暂无（锻造炉正在监视反复受挫的动议签名）")
        lines.append("锻造统计：成功锻造 %d 件｜成功序列样本 %d 条｜冷寂动议 %d 个签名"
                     % (e.forge.forged, len(e.forge.history),
                        len([1 for c in e.forge.fail_counts.values() if c > 0])))
        acts = dict(e.action_counts)
        if acts:
            lines.append("使用计数：" + "，".join(
                "%s×%d" % kv for kv in sorted(acts.items())))
        return nl.join(lines)

    # ---- 成长总结 ----
    def growth_summary(self) -> str:
        e = self.engine
        rows = e.metrics.rows
        nl = chr(10)
        lines = ["════ 成长总结（结构即智能的时间线）═══"]
        if not rows:
            lines.append("（采样不足：每 %d 拍生成一行成长日志，请再等等）" % 25)
            return nl.join(lines)
        q = max(1, len(rows) // 4)
        early, late = rows[:q], rows[-q:]

        def mean(rs, k):
            vs = [r[k] for r in rs if k in r]
            return sum(vs) / len(vs) if vs else 0.0

        comp_e, comp_l = mean(early, "competence"), mean(late, "competence")
        pop_e, pop_l = mean(early, "pop"), mean(late, "pop")
        bond_e, bond_l = mean(early, "bonds"), mean(late, "bonds")
        aro_e, aro_l = mean(early, "arousal"), mean(late, "arousal")
        nov_l = mean(late, "novelty")

        def arrow(a, b):
            return "↑" if b > a * 1.02 else ("↓" if b < a * 0.98 else "≈")

        lines += [
            "胜任度均值：早期 %.3f → 近期 %.3f %s（预测自己敏感通道的能力）"
            % (comp_e, comp_l, arrow(comp_e, comp_l)),
            "种群规模：早期 %.1f → 近期 %.1f %s（能量只流向让世界更可预测的结构）"
            % (pop_e, pop_l, arrow(pop_e, pop_l)),
            "键数量：早期 %.1f → 近期 %.1f %s｜唤醒度 %.3f → %.3f"
            % (bond_e, bond_l, arrow(bond_e, bond_l), aro_e, aro_l),
            "近期创新速率：每100拍变异 %.1f 次｜记忆重放累计 %d 次"
            % (nov_l, e.total_replays),
            "生命账本：出生 %d｜死亡 %d（淘汰由每个基元自己的账本决定）"
            % (e.births, e.deaths),
        ]
        ev = _all_events(e.p_events)
        first_bond = next((x for x in ev if x.get("type") == "BOND_FORMED"), None)
        first_mit = next((x for x in ev if x.get("type") == "MITOSIS"), None)
        tools_ev = [x for x in ev if x.get("type") == "TOOL_FORGED"]
        sc = [x for x in ev if x.get("type") == "SELF_CORRECTION"]
        corr = [x for x in ev if x.get("type") in ("CORRECTION", "PRAISE")]
        rs = [x for x in ev if x.get("type") == "PRIMORDIAL_RESEED"]
        if first_bond:
            lines.append("里程碑：首条键缔结于 tick %d（%s↔%s）"
                         % (first_bond["tick"], first_bond["a"], first_bond["b"]))
        if first_mit:
            lines.append("里程碑：首次有丝分裂于 tick %d（%s → %s）"
                         % (first_mit["tick"], first_mit["parent"], first_mit["child"]))
        if tools_ev:
            for t in tools_ev[-3:]:
                lines.append("里程碑：工具诞生 tick %d：%s = %s（亲代 %s）"
                             % (t["tick"], t["tool"], "→".join(t["recipe"]),
                                t["parent"]))
        if sc:
            lines.append("自我修正：%d 次（监察惊讶异常 ⇒ 回拢增益、上调阈值）" % len(sc))
        if corr:
            lines.append("人类纠正/表扬：%d 次，均已入约束册与问责窗" % len(corr))
        if rs:
            lines.append("原始汤补种：%d 次（诚实记录的边界条件）" % len(rs))
        return nl.join(lines)

    # ---- 自我描述 ----
    def self_description(self) -> str:
        e = self.engine
        nl = chr(10)
        kc = e.kinds_count()
        dom = max(kc, key=lambda k: kc[k])
        total_mut = sum(p.genome.mutations for p in e.population.values())
        oldest = min(e.population.values(), key=lambda p: p.genome.birth_tick)
        samples = e.describe_sample(2)
        lines = [
            "════ 我是谁 · 基元社会的自述 ═══",
            "我是 PRIMORDIA：一个没有中心的社会，此刻由 %d 个自治基元组成"
            "（sensor %d｜assoc %d｜memory %d｜meta %d｜effect %d）。"
            % (len(e.population), kc["sensor"], kc["assoc"], kc["memory"],
               kc["meta"], kc["effect"]),
            "我们之间没有层级，只有无偏均值聚合的共享场；沉默者的 0 与喧哗者的 1 "
            "以同一分母进入平均。",
            "占多数的是%s基元——他们说：\u201c%s\u201d" % (
                {"sensor": "感知", "assoc": "关联", "memory": "记忆",
                 "meta": "监察", "effect": "执行"}[dom],
                e.population and next(p.motto() for p in e.population.values()
                                      if p.kind == dom) or ""),
            "全群累计变异 %d 次；最年长者为 [%s]，生于 tick %d。"
            % (total_mut, oldest.pid, oldest.genome.birth_tick),
            "以下是最胜任的两名成员的自我刻画：",
        ]
        for s in samples:
            lines.append(s)
        if e.teachings:
            lines.append("观察者近来的教导：" +
                         "；".join(t["text"] for t in list(e.teachings)[-3:]))
        lines.append("我能自述以上一切，因为我的每一句话都来自可核对的账本。")
        return nl.join(lines)

    # ---- 事件尾 ----
    def events_tail(self, n: int = 8) -> str:
        ev = _tail_events(self.engine.p_events, None, n)
        if not ev:
            return "（事件流暂空）"
        out = []
        for x in reversed(ev):
            brief = _brief_event(x)
            out.append(brief)
        return chr(10).join(out)


def fld_str(fld: List[float]) -> str:
    return "[" + " ".join("%+.2f" % v for v in fld) + "]"


def _brief_event(x: Dict) -> str:
    t = x.get("type", "?")
    tk = x.get("tick", "?")
    if t == "BOND_FORMED":
        return "t%s 键+%s（%s↔%s）" % (tk, "", x.get("a"), x.get("b"))
    if t == "MITOSIS":
        return "t%s 分裂 %s→%s（%s）" % (tk, x.get("parent"), x.get("child"), x.get("kind"))
    if t == "DEATHS":
        return "t%s 死亡×%s" % (tk, x.get("n"))
    if t == "ACTION_EXECUTED":
        return "t%s 行动 %s Δprox=%s" % (tk, x.get("family"), x.get("prox_delta"))
    if t == "IRREVERSIBLE_BLOCKED":
        return "t%s 安全门拒绝不可逆动作" % tk
    if t == "MOTION_VETOED":
        return "t%s 动议否决 %s@%s" % (tk, x.get("family"), x.get("sig"))
    if t == "TOOL_FORGED":
        return "t%s 锻造 %s=%s" % (tk, x.get("tool"), "→".join(x.get("recipe", [])))
    if t == "SELF_CORRECTION":
        return "t%s 自我修正 err=%s" % (tk, x.get("err_ema"))
    if t in ("CORRECTION", "PRAISE"):
        return "t%s %s %s（触%d人）" % (tk, "表扬" if t == "PRAISE" else "纠正",
                                    x.get("family"), len(x.get("touched", [])))
    if t == "TEACH":
        return "t%s 教导注入：%s" % (tk, x.get("text", "")[:20])
    if t == "PRIMORDIAL_RESEED":
        return "t%s 原始汤补种×%s" % (tk, x.get("n"))
    if t == "DIFFERENTIATION":
        return "t%s 再分化 %s→%s" % (tk, x.get("pid"), x.get("to"))
    return "t%s %s" % (tk, t)


# ======================================================================
# HTML 仪表盘（零依赖：纯字符串拼 SVG）
# ======================================================================
def dashboard_html(engine: EvolutionEngine) -> str:
    r = Reporter(engine)
    f = engine.status_facts()
    rows = engine.metrics.rows
    comp = [x["competence"] for x in rows]
    pop = [x["pop"] for x in rows]
    aro = [x["arousal"] for x in rows]
    inc = [x["pool_income"] for x in rows]
    bnd = [x.get("bonds", 0) for x in rows]
    tls = [x.get("tools", 0) for x in rows]
    ev = _tail_events(engine.p_events, None, 14)
    ev_rows = "".join(
        "<tr><td>%s</td><td>%s</td></tr>" % (x.get("tick"), _brief_event(x))
        for x in reversed(ev))
    kinds = f["by_kind"]

    def esc(s):
        return str(s).replace("<", "&lt;").replace(">", "&gt;")
    html = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>PRIMORDIA v4 · 成长仪表盘</title><style>
body{font-family:'Segoe UI',system-ui,sans-serif;background:#10141c;color:#dfe7f2;margin:24px;}
h1{font-size:20px;} .grid{display:flex;flex-wrap:wrap;gap:18px;}
.card{background:#171d29;border:1px solid #232c3d;border-radius:10px;padding:14px 18px;min-width:300px;}
.kpi{font-size:26px;font-weight:600;color:#7fd1a8;margin:4px 0;}
table{border-collapse:collapse;font-size:13px;} td,th{padding:3px 10px;border-bottom:1px solid #232c3d;text-align:left;}
.chart{background:#11161f;border-radius:6px;} .muted{color:#77839a;}
</style></head><body>
<h1>PRIMORDIA v4 · 结构即智能 —— 成长仪表盘（tick TICKN）</h1>
<div class="grid">
<div class="card"><div>种群</div><div class="kpi">POPN</div>KINDTBL</div>
<div class="card"><div>胜任度（智能曲线）</div>SVGCOMP</div>
<div class="card"><div>集体唤醒度</div>SVGARO</div>
<div class="card"><div>环境收益</div>SVGINC</div>
<div class="card"><div>结构时间线：种群</div>SVGPOP</div>
<div class="card"><div>结构时间线：键数</div>SVGBND</div>
<div class="card"><div>能力边界：持有工具数</div>SVGTLS</div>
<div class="card"><h3>事件流尾</h3><table>EVROWS</table></div>
</div>
<p class="muted">数据源：out/growth_log.jsonl · out/events.jsonl · 每5秒自动刷新。<br>
最新报告：</p><pre style="white-space:pre-wrap">REPORT</pre>
</body></html>"""
    kindtbl = "<table>" + "".join(
        "<tr><td>%s</td><td>%d</td></tr>" % (k, v) for k, v in kinds.items()) + "</table>"
    html = (html.replace("TICKN", str(f["tick"])).replace("POPN", str(f["pop"]))
            .replace("KINDTBL", kindtbl)
            .replace("SVGCOMP", svg_sparkline(comp, color="#7fd1a8", label="competence"))
            .replace("SVGARO", svg_sparkline(aro, color="#7fb3ff", label="arousal"))
            .replace("SVGINC", svg_sparkline(inc, color="#f0c674", label="income"))
            .replace("SVGPOP", svg_sparkline(pop, color="#e58bd4", label="population"))
            .replace("SVGBND", svg_sparkline(bnd, color="#8fd3e8", label="bonds"))
            .replace("SVGTLS", svg_sparkline(tls, color="#c9e58b", label="tools"))
            .replace("EVROWS", ev_rows)
            .replace("REPORT", esc(r.status_report())))
    return html


def write_dashboard(engine: EvolutionEngine) -> str:
    path = os.path.join(engine.out_dir, "dashboard.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dashboard_html(engine))
    return path


# ======================================================================
# ANSI 快速看板
# ======================================================================
def ansi_board(engine: EvolutionEngine) -> str:
    rows = engine.metrics.rows
    nl = chr(10)
    f = engine.status_facts()
    lines = [
        "┌─ PRIMORDIA 看板 ─ tick %s ─ pop %s ─ bonds %s ─ tools %s ┐"
        % (f["tick"], f["pop"], f["bonds"],
           len([p for p in engine.population.values()
                if p.kind == "effect" and p.genome.tool])),
        "胜任度 " + sparkline([r["competence"] for r in rows]),
        "种群　 " + sparkline([r["pop"] for r in rows]),
        "唤醒度 " + sparkline([r["arousal"] for r in rows]),
        "收益　 " + sparkline([r["pool_income"] for r in rows]),
        "── 事件尾 ──",
    ]
    lines += ["  " + _brief_event(x) for x in _tail_events(engine.p_events, None, 5)]
    return nl.join(lines)


# ======================================================================
# Parser + 通信层
# ======================================================================
class CommunicationLayer:
    def __init__(self, engine: EvolutionEngine):
        self.engine = engine
        self.reporter = Reporter(engine)
        self.p_transcript = os.path.join(engine.out_dir, "transcript.jsonl")
        self.paused = False
        self.quit_flag = False
        self.pace_hint = 0          # +加速 / -减速 的档位提示
        self.interval = 0.25        # live 演化节拍（秒）

    # ---- 记录 ----
    def _tlog(self, direction: str, text: str) -> None:
        try:
            with open(self.p_transcript, "a", encoding="utf-8") as f:
                f.write(json.dumps({"tick": self.engine.tick,
                                    "dir": direction,
                                    "text": text}, ensure_ascii=False) + chr(10))
        except OSError:
            pass

    # ---- 主入口 ----
    def handle(self, text: str) -> str:
        text = (text or "").strip()
        self._tlog("human", text)
        resp = self._route(text)
        self._tlog("system", resp)
        return resp

    def _route(self, text: str) -> str:
        if not text:
            return self.HELP
        low = text.lower()
        if re.fullmatch(r"(帮助|指令|菜单|help|\?)", low):
            return self.HELP
        if re.search(r"报告|状态|体检", text):
            return self.reporter.status_report()
        if re.search(r"为什么|解释", text):
            return self.reporter.explain_last()
        if re.search(r"工具", text):
            return self.reporter.tools_report()
        if re.search(r"成长|演化史|历史|轨迹|进步", text):
            return self.reporter.growth_summary()
        if re.search(r"你是谁|自我描述|你是什么|介绍.{0,3}自己", text):
            return self.reporter.self_description()
        if re.search(r"看板|仪表", text):
            return ansi_board(self.engine)
        m = re.match(r"^教\s*[:：]\s*(.+)$", text)
        if m:
            info = self.engine.inject_teaching(m.group(1))
            return ("已把教导译成脉冲注入 %d 个感知基元的驱动项，"
                    "并将随场传播；文本已入记忆与事件流。" % info["sensors"])
        m = re.match(r"^纠正\s*[:：]\s*(\S+)\s*(.*)$", text)
        if m:
            res = self.engine.correct(m.group(1), good=False, reason=m.group(2))
            fam = res["family"] or "（未识别的动作名——已按最近行动处理）"
            return ("纠正已纳入演化方向：动作族 %s 在情境 %s 下记入约束册"
                    "（现生效 %d 条）；问责窗内 %d 名参与者已被扣能与敏化。"
                    % (fam, res["sig"] or "*", res["constraints_active"],
                       res["touched_n"]))
        m = re.match(r"^表扬\s*[:：]\s*(\S+)\s*(.*)$", text)
        if m:
            res = self.engine.correct(m.group(1), good=True, reason=m.group(2))
            fam = res["family"] or "（未识别的动作名——已按最近行动处理）"
            return ("表扬已纳入演化方向：动作族 %s 获得正反馈；问责窗内 %d 名参与者"
                    "获能量奖励与阈值下调；相关约束若存在已放宽一条。"
                    % (fam, res["touched_n"]))
        m = re.match(r"^(加速|减速)\s*(\d*)$", text)
        if m:
            step = int(m.group(2)) if m.group(2) else 1
            if m.group(1) == "加速":
                self.interval = max(0.03, self.interval * (0.5 ** step))
            else:
                self.interval = min(3.0, self.interval * (2 ** step))
            return "演化节奏已调整：每拍间隔 %.2f 秒。" % self.interval
        if re.fullmatch(r"(暂停|停)", text):
            self.paused = True
            return ("动力学已冻结（tick %s）。基元保持现状，我不会再推进任何一拍；"
                    "说『继续』即可恢复。" % self.engine.tick)
        if re.fullmatch(r"(继续|恢复)", text):
            self.paused = False
            return "动力学已恢复，从 tick %s 继续生长。" % self.engine.tick
        if re.fullmatch(r"(保存|存档)", text):
            p = self.engine.save_snapshot()
            return "状态快照已写入 %s（含 %d 个基元的完整基因组与账本）。" % (
                p, len(self.engine.population))
        if re.fullmatch(r"(退出|quit|exit)", low):
            self.quit_flag = True
            self.engine.save_snapshot()
            return "再见。快照已保存；我的全部成长痕迹都在 out/ 目录里，随时可以验尸。"
        if re.search(r"谢谢|辛苦", text):
            f = self.engine.status_facts()
            return ("不客气。此刻我仍活着：tick %s，%d 个基元，胜任度 %.3f。"
                    "我会继续思考。" % (f["tick"], f["pop"], f["competence"]))
        # 兜底：诚实告知 + 把自由文本当作环境低语注入
        info = self.engine.inject_teaching(text)
        f = self.engine.status_facts()
        return ("这不是我认识的指令，我不猜。你的话已作为环境低语注入 %d 个感知基元"
                "（当前唤醒度 %.3f）。可用指令见『帮助』。"
                % (info["sensors"], f["arousal"]))

    HELP = (
        "════ PRIMORDIA v4 · 通信协议 ═══" + chr(10) +
        "  报告／状态        全量量化状态报告" + chr(10) +
        "  为什么            解释最近一次集体行动的因果链" + chr(10) +
        "  工具              工具清单＋宏配方＋锻造统计" + chr(10) +
        "  成长              智能曲线对比＋结构时间线里程碑" + chr(10) +
        "  你是谁            种群自述" + chr(10) +
        "  教：<文本>        教导注入感知层" + chr(10) +
        "  纠正：<动作> [理由]   负校正（入约束册＋问责惩罚）" + chr(10) +
        "  表扬：<动作> [理由]   正校正" + chr(10) +
        "  加速／减速 [n]    演化节奏" + chr(10) +
        "  暂停／继续        冻结／恢复动力学" + chr(10) +
        "  看板              ANSI 实时仪表盘" + chr(10) +
        "  保存              写状态快照" + chr(10) +
        "  退出              存档并离开" + chr(10) +
        "其他任何文字都会作为环境低语注入感知层——我从不编造我没看见的事实。"
    )
