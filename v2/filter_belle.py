# -*- coding: utf-8 -*-
"""
SEED OS v2.2 — filter_belle.py
把 Belle 1M 中文指令数据过滤成语言皮层能消化的短问答库 corpus_large.tsv。

过滤规则（联想记忆吃不了任务型指令，只吃"一句问 → 一句答"）：
- input 字段为空（排除带上下文的多段任务）
- 问题/回答都不含换行、不超长（问 ≤ 40 字，答 ≤ 90 字）
- 排除内嵌素材的指令（含引号、"下面/以下/给定/例如/请将/请把"等任务模板词）
- 按归一化问题去重
"""
import json
import re

SRC = "Belle_open_source_1M.json"
DST = "corpus_large.tsv"

BAD_Q_WORDS = ("下面", "以下", "给定", "例如", "如下", "这段", "这篇", "这句",
               "请将", "请把", "将其", "把以下", "翻译成", "改写成", "缩写",
               "续写", "补全", "填空", "表格", "列表中", "编号", "选项")
BAD_A_START = ("“", "\"", "```", "1.", "2.", "-", "*")

_norm_re = re.compile(r"[\s，。？！,?!：:、.·~\"'（）()\[\]【】]+")


def norm(s):
    return _norm_re.sub("", s.lower())


def main():
    seen = set()
    kept, total = 0, 0
    with open(SRC, "r", encoding="utf-8") as fin, \
         open(DST, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (d.get("instruction") or "").strip()
            a = (d.get("output") or "").strip()
            inp = (d.get("input") or "").strip()
            if inp or not q or not a:
                continue
            if "\n" in q or "\n" in a or "\t" in q or "\t" in a:
                continue
            if len(q) > 45 or not (2 <= len(a) <= 120):
                continue
            if any(w in q for w in BAD_Q_WORDS):
                continue
            if any(a.startswith(c) for c in BAD_A_START):
                continue
            if q.count("？") + q.count("?") > 1:
                continue
            nq = norm(q)
            if not nq or nq in seen:
                continue
            seen.add(nq)
            fout.write(f"{q}\t{a}\n")
            kept += 1
    print(f"过滤完成：{total} 条原始 → {kept} 条短问答（保留率 {kept / total:.1%}）")


if __name__ == "__main__":
    main()
