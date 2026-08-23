# -*- coding: utf-8 -*-
"""
dashboard.py —— 成长仪表盘生成器
从 growth/growth_log.jsonl 与 growth/events.jsonl 生成静态 HTML+SVG：
种群、胜任度、世界可预测性、能量池、键数、唤醒度六条曲线 + 事件尾表。
纯标准库；python -m v3 模式或 run.py --mode report 调用。
"""
from __future__ import annotations

import json
import os


def build_dashboard(growthdir: str) -> str:
    rows = []
    p = os.path.join(growthdir, "growth_log.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    keys = [
        ("population", "种群"),
        ("competence", "平均胜任度"),
        ("predictability", "世界可预测性"),
        ("pool", "能量池"),
        ("bonds", "键数"),
        ("arousal", "集体唤醒度"),
    ]
    W, H, PAD = 760, 140, 30
    charts = []
    for key, title in keys:
        pts = [(r.get("tick", 0), r[key]) for r in rows
               if isinstance(r.get(key), (int, float))]
        if len(pts) < 2:
            continue
        x0 = pts[0][0]
        x1 = pts[-1][0]
        ys = [float(v) for _, v in pts]
        y0, y1 = min(ys), max(ys)
        rng = (y1 - y0) or 1.0
        span = max(1, x1 - x0)
        poly = " ".join(
            "%g,%g" % (PAD + (x - x0) / span * (W - 2 * PAD),
                       H - PAD - (yv - y0) / rng * (H - 2 * PAD))
            for x, yv in pts)
        charts.append(
            '<div class="card"><b>%s</b> &nbsp;min %.3g · max %.3g'
            '<br><svg viewBox="0 0 %d %d" preserveAspectRatio="none">'
            '<polyline points="%s" fill="none" stroke="#4ade80" '
            'stroke-width="2"/></svg></div>'
            % (title, y0, y1, W, H, poly))

    ev_rows = []
    evp = os.path.join(growthdir, "events.jsonl")
    if os.path.exists(evp):
        with open(evp, encoding="utf-8") as fh:
            tail = [ln for ln in fh.read().splitlines() if ln.strip()]
        for line in reversed(tail[-40:]):
            try:
                d = json.loads(line)
            except Exception:
                continue
            detail = ", ".join(
                "%s=%s" % (k, v) for k, v in d.items()
                if k not in ("tick", "type"))
            ev_rows.append("<tr><td>%s</td><td>%s</td><td>%s</td></tr>"
                           % (d.get("tick", ""), d.get("type", ""), detail))

    css = (
        "body{background:#0d1117;color:#c9d1d9;"
        "font-family:Consolas,monospace;margin:24px}"
        "h2{color:#58a6ff;font-size:18px}"
        ".card{border:1px solid #21262d;border-radius:8px;"
        "padding:8px 12px;margin:10px 0;background:#161b22}"
        "svg{width:100%;height:110px}"
        "table{font-size:12px;border-collapse:collapse}"
        "td{padding:2px 8px;border-bottom:1px solid #21262d}")

    html_parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>基元社会 · 成长仪表盘</title>",
        "<style>", css, "</style>",
        "<h2>基元社会 v3.1 · 成长仪表盘</h2>",
        "<p>快照 %d 行</p>" % len(rows),
    ]
    html_parts.extend(charts)
    html_parts.append(
        '<div class="card"><b>最近事件（新→旧，40 条）</b>'
        "<table>" + "".join(ev_rows) + "</table></div>")
    out = os.path.join(growthdir, "dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(html_parts))
    return out


if __name__ == "__main__":  # pragma: no cover
    print(build_dashboard("growth"))
