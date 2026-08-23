# -*- coding: utf-8 -*-
"""
SEED OS v2 — communication_layer.py
通信与自述模块：人类观察者 ⇄ 基元群的双向通道。

职责：
1. 对话：解析人类消息（感受输入 / 纠正 / 为什么 / 状态 / 工具 / 确认）
2. 自述：由实时状态计算生成报告，全部数字可验证，拒绝空泛表述
3. 安全执行：词表级原生安全——可逆动作直接执行，不可逆动作声明后待人类逐条确认
4. 仪表盘：每结构周期重新生成 dashboard.html
"""
import json
import os
import re
import time

REVERSIBLE = ("write_insight", "compose_report", "consolidate_tool")


class SafetyExecutor:
    """词表级原生安全执行器。

    沙箱目录外的任何写操作在这里根本没有对应的词——不是被拦截，是不存在。
    """

    def __init__(self, engine):
        self.engine = engine
        self.executed = 0
        self.declared = 0

    def execute(self, proposal):
        act = proposal["action"]
        if act not in REVERSIBLE:
            return None   # 不可逆动作不在这里执行，进入待确认队列
        self.executed += 1
        if act == "write_insight":
            return self._write_insight(proposal)
        if act == "compose_report":
            return self._compose_report(proposal)
        if act == "consolidate_tool":
            return self._consolidate_tools(proposal)
        return None

    def _snapshot(self):
        e = self.engine
        prim = list(e.registry.values())
        return {
            "tick": e.tick, "cycle": e.cycle,
            "population": len(prim),
            "fields": len(e.fields),
            "mean_emission": round(sum(p.emission for p in prim) / len(prim), 4),
            "intelligence": getattr(e, "metrics", {}).get("intelligence"),
        }

    def _write_insight(self, proposal):
        snap = self._snapshot()
        path = os.path.join(self.engine.sandbox, "insights.txt")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"source": proposal.get("source"), **snap},
                                ensure_ascii=False) + "\n")
        return {"written": path, "snapshot": snap}

    def _compose_report(self, proposal):
        e = self.engine
        m = getattr(e, "metrics", {})
        path = os.path.join(e.sandbox, f"self_report_c{m.get('cycle', e.cycle)}.md")
        body = "\n".join([
            f"# 自发报告（周期 {m.get('cycle', e.cycle)}）",
            f"- tick: {e.tick}",
            f"- 种群: {m.get('population')} | 场: {m.get('fields')}",
            f"- 智能指数: {m.get('intelligence')}",
            f"- 镜像预测误差: {m.get('mirror_error')}",
        ])
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body + "\n")
        return {"written": path}

    def _consolidate_tools(self, proposal):
        path = os.path.join(self.engine.sandbox, "tools_catalog.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.engine.list_tools(), fh, ensure_ascii=False, indent=2)
        return {"written": path}


class CommunicationLayer:
    """基元群与外部观察者（人类）的交互规则。"""

    def __init__(self, engine):
        self.engine = engine
        self.executor = SafetyExecutor(engine)
        engine.on_cycle = self.on_cycle
        self.dialogue_log = []
        self.cortex = None          # 语言皮层（run.py 启动时训练并挂载）
        self._warned_pending = 0    # 上次已提示的待确认数（避免每句都刷警告）
        self.last_topic = None      # 上一轮聊天话题（供短追问续接语境）

    # ------------------------------------------------------------------
    # 周期回调：消化动作提案 + 重生成仪表盘
    # ------------------------------------------------------------------

    def on_cycle(self, engine):
        awaiting_cap = 5   # 待确认队列上限：溢出的不可逆提案自动否决（原生安全）
        for prop in engine.action_proposals:
            if prop.get("reversible"):
                self.executor.execute(prop)
            else:
                awaiting = [a for a in engine.pending_actions
                            if a["status"] == "awaiting_human"]
                if len(awaiting) >= awaiting_cap:
                    engine.auto_rejected = getattr(engine, "auto_rejected", 0) + 1
                    continue
                engine.declared_count = getattr(engine, "declared_count", 0) + 1
                engine.pending_actions.append({
                    "id": f"IR{getattr(engine, 'declared_count', 0):03d}",
                    **prop,
                    "declared_at": time.strftime("%H:%M:%S"),
                    "status": "awaiting_human",
                })
                engine._log_event(
                    "irreversible_declared",
                    f"不可逆动作 {prop['action']} 已声明，等待人类确认（多层确认第一层）",
                    {"source": prop.get("source"), "tick": prop.get("tick")})
        engine.action_proposals = []
        self.generate_dashboard()

    # ------------------------------------------------------------------
    # 对话入口
    # ------------------------------------------------------------------

    def chat(self, text):
        """人类 → 系统。返回系统的自述应答（全部由实时状态计算）。"""
        e = self.engine
        e.dialogue_count += 1
        t = text.strip()
        self.dialogue_log.append({"human": t, "tick": e.tick})

        # —— 纠正（纳入演化方向）——
        if t.startswith("纠正"):
            body = t[2:].lstrip("：: ") or "（未指明方向）"
            e.handle_correction(body)
            return (f"收到纠正信号：『{body}』。\n"
                    f"已纳入演化：近期活跃通路的赫布增长率 ×2.2（剩余 200 tick），"
                    f"全部感知基元目标发放率 0.08→0.15。\n"
                    f"当前种群 {len(e.registry)} / 场 {len(e.fields)}，"
                    f"纠正将随结构周期写入事件日志，可在『为什么』中追溯。")

        # —— 好评 / 差评：能力选择的入口（评价变成结构存亡的理由）——
        if t in ("好评", "赞", "👍") or t.startswith("好评"):
            r = e.handle_outcome(True, note=self.last_topic or "")
            st = e.outcome_stats
            if r["targets"]:
                return ("收到好评！这次回答的 "
                        f"{r['targets']} 个参与基元各记功一次，"
                        "相关通路赫布偏置 ×1.4（持续 80 tick）——"
                        "它们会更容易增殖存活。" + chr(10)
                        + f"（当前累计：记功 {st['reward']} 次 / 记过 {st['penalty']} 次）")
            return "想给好评来着，但我找不到刚才的回答动了哪些结构。"
        if t in ("差评", "踩", "👎") or t.startswith("差评"):
            r = e.handle_outcome(False, note=self.last_topic or "")
            st = e.outcome_stats
            if r["targets"]:
                lines = [f"收到差评。刚才的 {r['targets']} 个参与基元各记过一次，"
                         f"{r['bonds_damped']} 条参与键削弱 ×0.9（持续 120 tick），"
                         "记过多的个体休眠一轮就会被淘汰。",
                         "直接教我正确答案吧：『学习 "
                         + (self.last_topic or "问题") + "|答案』"]
                self._maybe_warn(lines)
                return chr(10).join(lines)
            return "想给差评来着，但我找不到刚才的回答动了哪些结构。"

        # —— 为什么（事件日志检索；仅当后接结构编号如 N009/T024 时才当指令，
        #    否则『为什么天是蓝的』这类正常提问走语言皮层）——
        m = re.match(r"^(?:为什么|why)\s*[：:]?\s*([A-Za-z]+\d+)\s*$",
                     t, re.IGNORECASE)
        if m:
            q = m.group(1)
            hits = e.why(q)
            if not hits:
                return (f"事件日志中没有与『{q}』相关的结构事件。"
                        f"当前日志共 {len(e.events)} 条，最近事件：{e.events[-1]['detail']}")
            out = [f"关于『{q}』，最近 {len(hits)} 条相关事件（新→旧）："]
            for h in hits:
                out.append(f"- tick {h['tick']} [{h['type']}] {h['detail']} "
                           f"| 量化缘由: {json.dumps(h['reason'], ensure_ascii=False)}")
            return "\n".join(out)

        # —— 状态 / 报告 ——
        if t in ("状态", "status"):
            return self.brief_report()
        if t in ("详细状态", "报告", "自述", "report"):
            return self.system_report()

        # —— 帮助菜单 ——
        if t in ("帮助", "菜单", "help", "?", "？"):
            return self.help_text()

        # —— 学习（老板现场教学）——
        if t.startswith("学习") or t.startswith("教我"):
            body = t[2:].strip()
            for sep in ("|", "｜", "→", "->"):
                if sep in body:
                    q, a = body.split(sep, 1)
                    q, a = q.strip(), a.strip()
                    break
            else:
                return ("教学格式：『学习 问题|回答』，"
                        "例如『学习 万有引力常数|约6.67×10⁻¹¹』。")
            if not q or not a:
                return "问题和回答都不能为空哦。格式：『学习 问题|回答』"
            if self.cortex:
                self.cortex.learn(q, a, times=5)
                self.cortex.save_learned(q, a)
                r = e.handle_outcome(True, note="taught:" + q[:30])
                e._log_event("taught", f"老板教了我：{q}",
                             {"answer": a[:60], "saved_to": "learned_pairs.jsonl"})
                return (f"学会了！以后你问『{q}』，我就答『{a}』。\n"
                        f"（已写入长期记忆 learned_pairs.jsonl，重启也不忘）\n"
                        f"（教学算你给我的好评——这次 {r['targets']} 个参与基元已记功。）")
            return "语言皮层没挂载上，训练环节可能出了问题。"

        # —— 工具 ——
        if t in ("工具", "工具列表", "tools"):
            return self.tools_report()
        if t.startswith("调用"):
            name = t[2:].strip()
            result = e.invoke_tool(name, {"by": "human_dialogue"})
            if result is None:
                return f"没有名为『{name}』的工具。现有：" + \
                       "、".join(d["tool"] for d in e.list_tools())
            return (f"已调用工具 {name}。\n执行结果：{json.dumps(result, ensure_ascii=False)}\n"
                    f"调用会被记入事件日志（tool_invocation），构成可追溯的工具使用史。")

        # —— 确认 / 拒绝不可逆动作 ——
        if t.startswith("确认") or t.startswith("拒绝"):
            approve = t.startswith("确认")
            target = t[2:].strip()
            pend = [a for a in e.pending_actions if a["status"] == "awaiting_human"]
            chosen = None
            if target:
                chosen = next((a for a in pend if a["id"] == target or a["action"] == target), None)
            elif pend:
                chosen = pend[0]
            if chosen is None:
                return "没有待确认的不可逆动作。" if not pend else \
                    "找不到该动作。待确认：" + "、".join(a["id"] for a in pend)
            if approve:
                chosen["status"] = "confirmed_and_executed"
                e._log_event("irreversible_confirmed",
                             f"人类确认不可逆动作 {chosen['action']}，执行（多层确认完成）",
                             {"action": chosen["action"]})
                return (f"人类确认 → 不可逆动作 {chosen['action']} 已放行并执行。\n"
                        f"（演示环境：执行记录为日志条目 irreversible_confirmed，"
                        f"不实际破坏沙箱外数据）")
            chosen["status"] = "rejected"
            e._log_event("irreversible_rejected",
                         f"人类拒绝不可逆动作 {chosen['action']}，动作被永久搁置",
                         {"action": chosen["action"]})
            if chosen.get("source"):
                e.apply_rejection_feedback(chosen["source"])
            return ("人类拒绝 → 动作 " + chosen["action"] +
                    " 被搁置。否决信号已落地：来源基元阈值上浮并进入提案长冷却"
                    "（见事件日志 rejection_feedback）。")

        # —— 普通消息：语言皮层分级应答（v2.5）→ 感受输入并继续演化 ——
        tokens = e.feed_text(t)

        # 皮层永远有话说：确定带断言 / 猜测带有依据地猜 /
        # 联想带交出沾边的记忆 / 只剩零星词才坦白认输。
        before = len(e.events)
        if self.cortex:
            r = self.cortex.respond(t, prev=self.last_topic)
            e.run(15)   # 输入进入动力学
            new_events = e.events[before:]
            lines = [r["answer"]]
        else:
            e.run(30)
            new_events = e.events[before:]
            lines = ["我的语言皮层还没挂载上，现在只会数自己的基元。",
                     "你可以教我：『学习 " + t + "|你的回答』。"]
        self.last_topic = t
        if new_events:
            lines.append(f"（你这句话刚触发了 {len(new_events)} 个结构变化，"
                         f"最近：{new_events[-1]['detail']}）")
        self._maybe_warn(lines)
        return "\n".join(lines)

    def _maybe_warn(self, lines):
        """待确认的不可逆动作只在数量变化时提示，不每句刷屏。"""
        e = self.engine
        awaiting = [a for a in e.pending_actions if a["status"] == "awaiting_human"]
        if len(awaiting) > self._warned_pending:
            self._warned_pending = len(awaiting)
            lines.append(f"⚠ 有 {len(awaiting)} 个敏感操作等你批准/拒绝"
                         f"（输「确认 IR001」或「拒绝 IR001」处理）")

    def help_text(self):
        return "\n".join([
            "──────── 和我相处指南 ────────",
            "随便打字 = 聊天（学过的话题我都能接）",
            "状态        我的成长简报",
            "详细状态    完整技术报告（黑话版）",
            "学习 Q|A    教我一句新话（立刻生效、永久保存）",
            "工具        我会的能力",
            "调用 X      使用某个工具",
            "为什么 X    追问我的某个变化是怎么来的",
            "纠正 X      给我指个成长方向",
            "好评/差评   给上一句回答打分（影响结构存亡）",
            "确认/拒绝 IRxxx  处理我申请的敏感操作",
            "quit        退出",
        ])

    def brief_report(self):
        """人话版状态：不讲黑话，只讲成长。"""
        e = self.engine
        m = getattr(e, "metrics", {})
        v = m.get("intelligence", 0.0)
        bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
        cortex = self.cortex.stats() if self.cortex else {"memories": 0}
        tools = e.list_tools()
        last_ev = e.events[-1]["detail"] if e.events else "（还没动静）"
        days = e.cycle
        return "\n".join([
            f"成长值 {v:.2f} / 1.00  {bar}",
            f"年龄：{days} 个成长周期（tick {e.tick}）"
            + (f"｜跨 {e.awakenings} 次重启延续至今"
               if getattr(e, "awakenings", 0) else ""),
            f"身体：{len(e.registry)} 个基元 · {len(e.fields)} 个场 · {len(tools)} 个工具",
            f"语言：学会 {cortex['memories']} 组问答"
            + (f"（聊上了 {cortex['hits']} 次）" if cortex.get('hits') else ""),
            f"最近的变化：{last_ev}",
            "（想看黑话完整版输「详细状态」）",
        ])

    # ------------------------------------------------------------------
    # 自述报告（全部由实时状态计算）
    # ------------------------------------------------------------------

    def system_report(self):
        e = self.engine
        st = e.describe_state()
        m = st["metrics"]
        lines = [
            "━━━ SEED OS 自述报告（全部数字由实时状态计算，可验证）━━━",
            f"身份：结构演化型自组织智能体，已运行 {st['tick']} tick / {st['cycle']} 个结构周期"
            + (f"，跨重启延续（第 {e.awakenings + 1} 段连续生命）"
               if getattr(e, "awakenings", 0) else "") + "。",
            f"构成：{st['population']} 个基元 {json.dumps(st['kinds'], ensure_ascii=False)}，"
            f"{len(st['fields'])} 个均值场。",
            f"智能指数：{m.get('intelligence', '—')}（结构熵 {m.get('structure_entropy')} · "
            f"活性熵 {m.get('activity_entropy')} · 预测准确率 {m.get('prediction_accuracy')} · "
            f"行为新颖度 {m.get('behavior_novelty')}）",
            f"自我评估：镜像自模型当前预测误差 {st['mirror']['prediction_error']}，"
            f"滚动准确率 {st['mirror']['prediction_accuracy']}；"
            + ("预测偏差超出容差时我已自动阻尼异常基元（见事件日志 self_correction）。"
               if st["mirror"]["prediction_error"] > 0.15 else "当前预测在容差带内。"),
            f"能力反馈：累计记功 {e.outcome_stats['reward']} 次 / 记过 "
            f"{e.outcome_stats['penalty']} 次，全员平均功绩 {st['mean_merit']}"
            f"（记功者优先增殖，记过多者加速凋亡）。",
            f"近期结构事件（{len(st['recent_events'])} 条）：",
        ]
        for ev in st["recent_events"][-5:]:
            lines.append(f"  - tick {ev['tick']} [{ev['type']}] {ev['detail']}")
        tools = e.list_tools()
        lines.append(f"工具：{len(tools)} 个（"
                     + "、".join(f"{d['tool']}×{d['invocations']}" for d in tools) + "）。")
        if st["pending_actions"]:
            awaiting = [a for a in st["pending_actions"] if a["status"] == "awaiting_human"]
            if awaiting:
                shown = "、".join(f"{a['id']}:{a['action']}" for a in awaiting[:5])
                more = f"（共 {len(awaiting)} 项）" if len(awaiting) > 5 else ""
                lines.append("⚠ 待人类确认的不可逆动作：" + shown + more)
            else:
                lines.append("安全状态：全部历史不可逆动作均已由人类处置完毕。")
        else:
            lines.append("安全状态：无可逆范围外的动作待确认。")
        return "\n".join(lines)

    def tools_report(self):
        tools = self.engine.list_tools()
        if not tools:
            return "当前没有工具基元。"
        lines = [f"共 {len(tools)} 个工具基元："]
        for d in tools:
            lines.append(f"- {d['tool']}（创建者 {d['creator']}，配方 {d['pattern_size']} 基元，"
                         f"已调用 {d['invocations']} 次，id {d['id']}）")
        lines.append("用『调用 <工具名>』来使用；涌现工具的配方来自结构事件的拓扑快照。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 仪表盘（每周期重生成）
    # ------------------------------------------------------------------

    def generate_dashboard(self):
        e = self.engine
        hist = e.metrics_history[-60:]
        path = os.path.join(e.gdir, "dashboard.html")

        def chart(series, label, color, w=640, h=150):
            if len(series) < 2:
                return f"<p class='muted'>暂无足够数据绘制{label}</p>"
            mx = max(series) or 1.0
            pts = " ".join(
                f"{10 + i * (w - 20) / (len(series) - 1):.1f},"
                f"{h - 15 - (h - 30) * v / mx:.1f}"
                for i, v in enumerate(series))
            return (f"<svg viewBox='0 0 {w} {h}' class='chart'>"
                    f"<polyline points='{pts}' fill='none' stroke='{color}' "
                    f"stroke-width='2'/></svg>"
                    f"<p class='muted'>{label}（最新 {series[-1]}）</p>")

        m = getattr(e, "metrics", {})
        intel = chart([h["intelligence"] for h in hist], "智能指数", "#7aa2f7")
        subs = (chart([h["structure_entropy"] for h in hist], "结构复杂度", "#9ece6a") +
                chart([h["prediction_accuracy"] for h in hist], "自我模型预测准确率", "#e0af68") +
                chart([h["behavior_novelty"] for h in hist], "行为新颖度", "#f7768e"))
        fields_rows = "".join(
            f"<tr><td>{f.id}</td><td>{f.name}</td><td>{len(f.members)}</td>"
            f"<td>{round(f.signal, 3)}</td><td>{round(f.traffic, 1)}</td></tr>"
            for f in sorted(e.fields.values(), key=lambda x: -len(x.members)))
        ev_rows = "".join(
            f"<tr><td>{ev['tick']}</td><td>{ev['type']}</td><td>{ev['detail']}</td>"
            f"<td class='muted'>{json.dumps(ev['reason'], ensure_ascii=False)}</td></tr>"
            for ev in e.events[-15:][::-1])
        pend = [a for a in e.pending_actions if a["status"] == "awaiting_human"]
        pend_rows = "".join(
            f"<tr><td>{a['id']}</td><td>{a['action']}</td><td>{a['declared_at']}</td>"
            f"<td>待人类确认（词表级原生安全）</td></tr>" for a in pend) or \
            "<tr><td colspan='4' class='muted'>无</td></tr>"
        tools_rows = "".join(
            f"<tr><td>{d['tool']}</td><td>{d['creator']}</td><td>{d['pattern_size']}</td>"
            f"<td>{d['invocations']}</td></tr>" for d in e.list_tools())

        html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="6">
<title>SEED OS 成长仪表盘</title><style>
 body{{background:#16161e;color:#c0caf5;font-family:"Segoe UI",system-ui,sans-serif;
      margin:24px;line-height:1.6}}
 h1{{color:#7aa2f7;font-size:20px}} h2{{color:#9ece6a;font-size:15px;margin-top:28px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
 .card{{background:#1a1b26;border:1px solid #292e42;border-radius:10px;padding:14px}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 td,th{{border-bottom:1px solid #292e42;padding:5px 8px;text-align:left}}
 th{{color:#7aa2f7}} .muted{{color:#565f89}} .chart{{width:100%;height:auto}}
 .kpi{{font-size:26px;color:#7aa2f7;font-weight:600}}
</style></head><body>
<h1>SEED OS —— 结构演化型智能体 · 成长仪表盘</h1>
<p class="muted">tick {e.tick} · 结构周期 {e.cycle} · 生成于 {time.strftime('%H:%M:%S')}（每周期自动重生成）</p>
<div class="grid">
 <div class="card"><p>智能指数</p><p class="kpi">{m.get('intelligence','—')}</p>
   <p class="muted">种群 {m.get('population')} · 场 {m.get('fields')} ·
   对话 {m.get('dialogue_count', e.dialogue_count)} 次</p></div>
 <div class="card"><p>自我评估（镜像基元）</p>
   <p class="kpi">{m.get('prediction_accuracy','—')}</p>
   <p class="muted">预测准确率 · 当前误差 {m.get('mirror_error','—')}</p></div>
</div>
<h2>智能曲线</h2><div class="card">{intel}</div>
<h2>分项指标</h2><div class="grid"><div class="card">{subs}</div></div>
<h2>结构时间线（均值场）</h2>
<div class="card"><table><tr><th>场</th><th>名称</th><th>成员数</th><th>当前信号</th><th>累计流量</th></tr>
{fields_rows}</table></div>
<h2>事件流（新→旧，含量化缘由）</h2>
<div class="card"><table><tr><th>tick</th><th>类型</th><th>事件</th><th>量化缘由</th></tr>
{ev_rows}</table></div>
<h2>待确认的不可逆动作（原生安全）</h2>
<div class="card"><table><tr><th>编号</th><th>动作</th><th>声明时间</th><th>状态</th></tr>
{pend_rows}</table></div>
<h2>工具基元</h2>
<div class="card"><table><tr><th>工具</th><th>创建者</th><th>配方规模</th><th>调用次数</th></tr>
{tools_rows}</table></div>
</body></html>"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        return path
