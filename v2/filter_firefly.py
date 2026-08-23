# -*- coding: utf-8 -*-
"""firefly-train-1.1M.jsonl → 过滤出适合联想记忆的短问答，合并进 corpus_large.tsv。

筛选标准（延续 v2.2 的哲学：只喂"问题和回答都像人话"的数据）：
- kind 白名单：OpenQA / 百科问答类优先
- 或 input 以问号结尾且无任务前缀（"："命令式开头的排除）
- 问题 ≤60 字、回答 ≤120 字
- 与现有 corpus_large.tsv 按归一化问题去重
"""
import json
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

SRC = "firefly-train-1.1M.jsonl"
DST = "corpus_large.tsv"

# firefly 的 kind 种类很多，这些是天然问答：
QA_KINDS = {"OpenQA", "BaiKeQA", "ZhihuQA", "MedicalQA", "KeBaiKe"}
# 明确的任务型 kind，直接排除：
BAD_KINDS = {"NLI", "Summary", "ParagraphSummary", "NewsSummary", "Couplet",
             "MusicComment", "ClassicalPoem", "ChineseModernPoetry",
             "Translation", "NER", "TextCorrection", "CPL",
             "KeyWordRecognition", "Sentiment", "ASR", "TTS", "STT"}

# 任务指令前缀特征（"自然语言推理：" "输出摘要：" 之类）
TASK_PREFIX = re.compile(r"^[^\n]{1,12}[：:]\s*\n|^翻译|^改写|^续写|^仿写|^摘要|^对联")
QMARK = re.compile(r"[?？]$")


def extract_openqa(text: str):
    """从 firefly OpenQA 模板里提取真正的问题。

    模板形如：
        请回答下面的问题：\n<问题>\n答案：
        请回答下面的问题：\n主题：<主题>\n 描述：<描述>\n答案：
        请回答问题：<问题>
    """
    t = text.strip()
    # 去掉开头模板语
    t = re.sub(r"^请回答下面的问题[：:]?\s*\n?", "", t)
    t = re.sub(r"^请回答问题[：:]?\s*", "", t)
    # 去掉结尾"答案："
    t = re.sub(r"\n?\s*答案[：:]?\s*$", "", t)
    # 主题/描述格式：主题行当问题
    m = re.search(r"主题[：:]\s*(.+)", t)
    if m:
        return m.group(1).strip()
    # 多行时取最后一行（模板语后面的问题本体）
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def norm_q(q: str) -> str:
    return re.sub(r"[\s，。？！?!.,\"'“”‘’：:、]", "", q)


def main():
    # 现有语料的问题集合（去重基准）
    seen = set()
    kept_lines = []
    with open(DST, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                seen.add(norm_q(parts[0]))
                kept_lines.append(line)
    n_old = len(kept_lines)

    kind_stat = Counter()
    added = 0
    with open(SRC, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = d.get("kind", "")
            q = (d.get("input") or "").strip()
            a = (d.get("target") or "").strip()
            kind_stat[kind] += 1
            if kind in BAD_KINDS:
                continue
            # OpenQA：从模板里提取真问题
            if kind == "OpenQA":
                q = extract_openqa(q)
            if not q or not a:
                continue
            if "\t" in q or "\t" in a or "\n" in q or "\n" in a:
                continue
            # 问题形态：白名单 kind 或 问号结尾
            if kind not in QA_KINDS and not QMARK.search(q):
                continue
            # 排除任务指令
            if TASK_PREFIX.match(q):
                continue
            if len(q) > 60 or not (2 <= len(a) <= 200):
                continue
            k = norm_q(q)
            if k in seen:
                continue
            seen.add(k)
            kept_lines.append(f"{q}\t{a}\n")
            added += 1

    with open(DST, "w", encoding="utf-8") as fh:
        fh.writelines(kept_lines)

    print(f"firefly 全量 kind 分布（前12）:")
    for k, n in kind_stat.most_common(12):
        print(f"  {k}: {n}")
    print(f"原有 {n_old} 组 + firefly 新增 {added} 组 = 共 {len(kept_lines)} 组")
    import os
    print(f"corpus_large.tsv 大小: {os.path.getsize(DST)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
