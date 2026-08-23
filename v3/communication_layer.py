# -*- coding: utf-8 -*-
"""
communication_layer.py —— 透明沟通层：人 ↔ 基元社会
==========================================================
Reporter 是架在统计事实上的一块单向玻璃：
一切对外语句均由模板从 MetricsRegistry 的量化事实即时渲染，
每个数字都可对照 growth/growth_log.jsonl 验证——拒绝空泛表述。

协议文法（中文优先）：
  报告 / 状态 / 体检            → 全量状态报告（带数字）
  为什么 / 解释                 → 最近一次集体行动的因果链
  纠正：<动作> 不好(:理由)       → 负校正入册 + 参与者阈值上调
  表扬：<动作> 好               → 正强化：阈值下调 + 能量奖励
  教：<文本>                    → 存入教导册，记忆基元获能量
  工具                          → 工具清单（含锻造出的宏工具）
  重置环境                      → 发起不可逆请求（多层确认）
  确认 <令牌>                   → 提供人类确认令牌
  加速 n / 减速 n               → 调整演示节奏
  暂停 / 继续                   → 冻结 / 恢复动力学
  帮助                          → 协议全文
纯标准库。
"""
from __future__ import annotations

import re
from typing import Dict, Optional

ACTION_ALIAS = {
    "探测": "probe", "探查": "probe",
    "正推": "nudge_plus", "正推环境": "nudge_plus", "正向轻推": "nudge_plus",
    "负推": "nudge_minus", "负向轻推": "nudge_minus",
    "重置": "reset", "清零": "reset",
    "any": "any", "*": "any", "全部": "any", "任何": "any",
}

GOOD_WORDS = ("好", "对", "棒", "不错", "赞")
BAD_WORDS = ("不好", "错", "禁止", "别再", "不要")


def _norm_action(word: str) -> str:
    w = (word or "").strip().lower()
    return ACTION_ALIAS.get(w, ACTION_ALIAS.get(w.lower(), w))


def _fmt(v) -> str:
    if isinstance(v, float):
        s = f"{v:.4g}"
        return s
    return str(v)


# ----------------------------------------------------------------------
# 解析器：把人类输入解析为结构化意图
# ----------------------------------------------------------------------
def parse(text: str) -> Dict:
    t = (text or "").strip()
    if not t:
        return {"type": "EMPTY"}

    # ---- 不可逆确认 ----
    m = re.match(r"^(?:确认|confirm)\s*[:：]?\s*(\S+)?\s*$", t, re.I)
    if m:
        return {"type": "CONFIRM", "token": m.group(1) or ""}
    m = re.match(r"^(?:重置环境|清零环境|执行重置)$", t)
    if m:
        return {"type": "IRREV_REQUEST", "tool": "reset"}

    # ---- 纠正 / 表扬 ----
    m = re.match(r"^(纠正|表扬|禁止)\s*[:：]?\s*(\S+)?\s*(.*)$", t)
    if m:
        verb, word, rest = m.group(1), m.group(2) or "", m.group(3) or ""
        positive = verb == "表扬"
        if not positive:
            if any(g in rest or g in word for g in GOOD_WORDS) and \
                    not any(b in rest for b in BAD_WORDS):
                positive = True
        note = rest.strip(" ：:，。!！?")
        target = _norm_action(word) if word else "any"
        return {"type": "CORRECT", "target": target,
                "positive": positive, "note": note}

    # ---- 教导 ----
    m = re.match(r"^(?:教|教导|请记)\s*[:：]\s*(.+)$", t)
    if m:
        return {"type": "TEACH", "text": m.group(1).strip()}

    # ---- 节奏 / 暂停 ----
    m = re.match(r"^(加速|减速)\s*(\d+)?\s*$", t)
    if m:
        d = int(m.group(2) or 5)
        return {"type": "SPEED", "delta": d if m.group(1) == "加速" else -d}
    if re.match(r"^(暂停|停止|停)\s*$", t):
        return {"type": "PAUSE"}
    if re.match(r"^(继续|恢复)\s*$", t):
        return {"type": "RESUME"}

    # ---- 简单意图 ----
    if re.search(r"(报告|状态|体检)", t):
        return {"type": "REPORT"}
    if re.search(r"(为什么|解释|原因)", t):
        return {"type": "EXPLAIN"}
    if re.search(r"工具", t):
        return {"type": "TOOLS"}
    if re.search(r"(词汇|词表|学会的词|语义)", t):
        return {"type": "LEXICON"}
    if re.search(r"(帮助|^指令$|你能做什么)", t):
        return {"type": "HELP"}
    if re.search(r"(你是谁|你是什么|你好|在吗)", t):
        return {"type": "CHAT"}
    return {"type": "UNKNOWN", "text": t}


# ----------------------------------------------------------------------
# 渲染器：模板 + 事实
# ----------------------------------------------------------------------
def render_report(f: Dict) -> str:
    L = []
    L.append(f"═══ 群落状态报告 · 第 {f['tick']} 拍 ═══")
    bk = f["by_kind"]
    L.append(
        f"◆ 种群 {f['population']} 体"
        f"（感知{bk.get('sensor',0)} 关联{bk.get('assoc',0)} "
        f"记忆{bk.get('memory',0)} 监察{bk.get('meta',0)} "
        f"执行{bk.get('effect',0)}）；键 {f['bonds']} 条；"
        f"连通片 {f['components']} 个；平均度 {f['mean_degree']}")
    L.append(f"◆ 能量池 {f['pool']}")
    L.append(f"◆ 环境 x={f['env_x']}｜世界可预测性 {f['predictability']}"
             f"（这是群落唯一收入来源的系数）")
    L.append(f"◆ 平均胜任度 {f['competence']}｜新生比例 {f['novelty']}"
             f"｜同步度 {f['coherence']}｜唤醒度 {f['arousal']}")
    c = (f"◆ 累计：出生 {f['births_total']}｜死亡 {f['deaths_total']}"
         f"｜集体行动 {f['actions_total']}｜变异 {f['mutations_total']}"
         f"｜纠正 {f['corrections_total']}｜工具诞生 "
         f"{f['tools_born_total']}｜约束在册 {f['constraints_n']}")
    L.append(c)
    L.append(f"◆ 世系：最深 {f['lineage_max']} 代｜平均 "
             f"{f['lineage_avg']} 代（有丝分裂的谱系痕迹）")
    if f.get("lexicon_n"):
        ln = f["lexicon_nearest"]
        near = (f"；当前情境最近词「{ln['word']}」（相似度 {ln['sim']}）"
                if ln else "；当前情境暂无近亲词")
        L.append(f"◆ 语义萌芽：已绑定 {f['lexicon_n']} 词{near}")
    sp = f["sparks"]
    L.append(f"◆ 种群曲线 {sp['population']}")
    L.append(f"◆ 胜任度曲线 {sp['competence']}")
    ev = f["recent_events"]
    if ev:
        L.append("◆ 最近事件：")
        for e in ev[:5]:
            L.append(f"   {e}")
    return "\n".join(L)


def render_explain(f: Dict) -> str:
    a = f.get("last_action")
    if not a:
        return ("我还没有执行过任何集体行动。" 
                f"当前第 {f['tick']} 拍，种群 {f['population']}，"
                "唤醒度达到触发线后才会表决。")
    steps = "+".join(a["steps"]) if a["steps"] else "(被安全门拦截)"
    L = [f"── 最近一次行动解释 ──",
         f"第 {a['tick']} 拍，情境 {a['context']}：集体唤醒度"
         f"（近4拍内发放过的基元占比）为 {a['arousal']}，"
         "越过法定线后开始表决。",
         f"执行基元 {a['actor']} 以支持度 {a['support']}"
         f"（领先第二名 {a['margin']}）胜出，执行 [{steps}]。",
         f"行动前后世界可预测性变化 {a['quality']:+.4f}，"
         f"据此向参与联盟均分奖励 {a['reward']}。",
         f"联盟成员 {len(a['coalition'])} 名"
         f"（近6拍内共同发放者）：{'、'.join(a['coalition'][:8])}"
         f"{'…' if len(a['coalition']) > 8 else ''}。"]
    if a["tool"] and a["tool"].startswith("macro:"):
        L.append(f"注意：该动作为锻造炉诞生的宏工具 {a['tool'][6:]}。")
    return "\n".join(L)


def render_tools(f: Dict) -> str:
    L = ["─── 工具清单 ───"]
    for tl in f["tools"]:
        rev = "可逆" if tl["reversible"] else "不可逆(需多层确认)"
        L.append(f"· {tl['name']} —— {tl['desc']}【{rev}】")
    if f["macros"]:
        L.append(f"── 锻造炉产物（共{len(f['macros'])}件，尝试"
                 f"{f['forge_attempts']}次）──")
        for m in f["macros"]:
            L.append(f"· {m['name']} = {m['recipe']}（诞生于第"
                     f"{m['born_tick']}拍）")
    else:
        L.append(f"── 锻造炉尚未产出宏工具（尝试 {f['forge_attempts']} 次，"
                 f"冷寂签名 {f['cold_sigs']} 个）──")
    L.append(f"约束库在册 {f['constraints_n']} 条；教导册 "
             f"{f['teachings_n']} 条。")
    return "\n".join(L)


def render_correction(res: Dict, positive: bool, target: str) -> str:
    head = "已收到表扬" if positive else "已收到纠正"
    hit = "并匹配到最近一次同类行动" if res["matched"] else           "（近期无同类行动，约束已先行入册，命中即生效）"
    L = [f"{head}：【{target}】{hit}。"]
    if res["affected"]:
        names = "、".join(res["affected"][:8])
        more = "…" if len(res["affected"]) > 8 else ""
        L.append(f"涉事参与者 {len(res['affected'])} 名（{names}{more}）：")
        sign = "下调" if positive else "上调"
        L.append(f"  · 阈值{sign} {abs(res['delta_theta'])}；")
        if positive:
            L.append(f"  · 每人能量奖励 {res['bonus']}。")
        else:
            L.append(f"  · 该情境已写入约束库"
                     f"（context={res['context']}），"
                     "此后同类动议在该情境中将丧失合格资格。")
    else:
        if not positive:
            L.append(f"  · 约束已入册（context={res.get('context','*')}）。")
    L.append("此纠正已成为本地事实，其后果交由自然法则放大或遗忘。")
    return "\n".join(L)


def render_lexicon(f: Dict) -> str:
    L = [f"─── 语义萌芽 · 已绑定 {f['lexicon_n']} 词 ───"]
    if not f["lexicon"]:
        L.append("还没有词。用「教：这个区域很危险」教我——"
                 "我会在当下经验上命名它，之后相似情境我会认得。")
        return "\n".join(L)
    for e in f["lexicon"]:
        L.append(f"· 「{e['word']}」 绑定于第{e['bound_tick']}拍，"
                 f"被引用 {e['uses']} 次")
    ln = f.get("lexicon_nearest")
    if ln:
        L.append(f"当前场模式最近词：「{ln['word']}」（余弦相似度 "
                 f"{ln['sim']}）——词与经验的距离是可验证的数字。")
    return "\n".join(L)


HELP_TEXT = """─── 通信协议 ───
报告 / 状态 / 体检      全量状态报告（全部带可验证数字）
为什么 / 解释           最近一次集体行动的因果链
纠正：<动作> 不好：理由  负校正：约束入册+参与者变迟钝
表扬：<动作> 好         正强化：参与者敏化+能量奖励
  （动作可用：probe/nudge_plus/nudge_minus/reset/any）
教：<文本>              存入教导册
工具                    工具清单与锻造产物
重置环境                发起不可逆请求 → 需监察2/3清醒 + 你的令牌
确认 <令牌≥4字符>       提供人类确认令牌
加速 n / 减速 n         调整节奏；暂停 / 继续
帮助                    显示本协议"""

CHAT_TEXT = ("我是基元社会：{}个自治基元经由共享场的无偏均值聚合耦合，"
             "在五条自然法则下自我演化。我没有中央控制器，"
             "也没有预装的语言模型——你现在读到的每个字都来自统计事实模板。"
             "当前第{}拍，我已存在{}拍。")


# ----------------------------------------------------------------------
# 通信层：路由意图 → 引擎 API → 渲染回复
# ----------------------------------------------------------------------
class CommunicationLayer:
    def __init__(self, engine):
        self.engine = engine
        self.j = engine.journal
        self.turns = 0

    def dialogue(self, text: str) -> str:
        self.j.dialogue("human", text)
        intent = parse(text)
        try:
            reply = self._dispatch(intent)
        except Exception as ex:            # 通信层绝不拖垮演化主循环
            reply = f"[通信层内部错误已隔离] {ex}"
        self.turns += 1
        self.j.dialogue("agi", reply)
        return reply

    # ---- 分发 ----
    def _dispatch(self, it: Dict) -> str:
        ty = it["type"]
        eng = self.engine
        if ty in ("REPORT", "EMPTY"):
            return render_report(eng.introspect())
        if ty == "EXPLAIN":
            return render_explain(eng.introspect())
        if ty == "TOOLS":
            return render_tools(eng.introspect())
        if ty == "CORRECT":
            res = eng.apply_correction(it["target"], it["positive"],
                                       it.get("note", ""))
            return render_correction(res, it["positive"],
                                     it["target"])
        if ty == "TEACH":
            r = eng.teach(it["text"])
            base = (f"已存入教导册（累计 {r['stored']} 条），"
                    f"{r['memories_touched']} 个记忆基元分得能量。")
            if r.get("word"):
                base += (f"\n语义萌芽：我把「{r['word']}」绑定到当下的场模式"
                         "（第" + str(eng.t) + "拍的经验）。"
                         "以后相似情境我会认得它。输入「词汇」查看词册。")
            else:
                base += "教导以脉冲方式影响场，其命运由自然法则决定。"
            return base
        if ty == "LEXICON":
            return render_lexicon(eng.introspect())
        if ty == "IRREV_REQUEST":
            return self._irrev_request(it["tool"])
        if ty == "CONFIRM":
            return self._irrev_confirm(it["token"])
        if ty == "SPEED":
            d = it["delta"]
            self.speed_factor = getattr(self, "speed_factor", 1.0)
            self.speed_factor = max(0.05, min(20.0, self.speed_factor *
                                              (1.25 if d > 0 else 0.8)))
            return f"节奏已调整 ×{self.speed_factor:.2f}（拍间隔相应变化）。"
        if ty == "PAUSE":
            eng.paused = True
            return (f"动力学已冻结于第 {eng.t} 拍"
                    f"（种群 {len(eng.population)}）。输入「继续」恢复。")
        if ty == "RESUME":
            eng.paused = False
            return f"动力学已从第 {eng.t} 拍恢复。"
        if ty == "HELP":
            return HELP_TEXT
        if ty == "CHAT":
            return CHAT_TEXT.format(len(eng.population),
                                    eng.t, eng.t)
        # UNKNOWN：诚实相告
        return ("未能解析该输入。可用指令如下（或输入「帮助」）：\n"
                "报告｜为什么｜纠正：X 不好｜表扬：X 好｜教：文本｜工具｜"
                "重置环境｜确认 令牌｜加速/减速｜暂停/继续")

    # ---- 不可逆管线 ----
    def _irrev_request(self, tool: str) -> str:
        r = self.engine.irreversible_request(tool)
        if r.get("ok"):
            ratio = r.get("meta_ratio", "?")
            return (f"不可逆请求【{tool}】已通过监察法定线"
                    f"（清醒率 {ratio}）。\n这是不可逆动作，需要你的显式令牌：\n"
                    "  输入：确认 <任意≥4字符令牌>\n"
                    "不提供令牌即视为否决。")
        return "请求被挡下：%s" % r.get("why", "未知")

    def _irrev_confirm(self, token: str) -> str:
        pend = self.engine.pending_irreversible
        if not pend or not pend.get("meta_ok"):
            return "当前没有等待人类令牌的不可逆请求。"
        r = self.engine.irreversible_confirm(token)
        if r.get("ok"):
            steps = "+".join(s.get("op", "?") for s in r.get("steps", []))
            return (f"多层确认完成，执行基元 {r['actor']} 已执行"
                    f" [{steps}]。全过程见 events.jsonl。")
        return "确认未通过：%s" % r.get("why", "未知")


def periodic_report(layer: CommunicationLayer) -> str:
    return render_report(layer.engine.introspect())
