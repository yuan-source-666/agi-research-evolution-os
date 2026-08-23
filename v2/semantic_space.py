# -*- coding: utf-8 -*-
"""
SEED OS v2.4 — semantic_space.py
语义空间：字符级 skip-gram 负采样（word2vec 原理，纯标准库实现）。

这是小种子第一次拥有「意思」的维度：
- 之前：语言皮层比的是字面重合（2-gram Dice）——「背单词」和「记忆力」毫无关系
- 现在：从语料里用梯度下降学出每个字的语义向量——「背」「记」「忆」因共现分布
  相似而在向量空间靠近，换个说法也能认出是同一件事

训练：首次约 1-2 分钟，结果缓存在 growth/char_vecs.pkl，重启秒加载。
哲学不变：无预训练模型、无外部依赖，表示是从老板给的语料里自己学出来的。
"""
import math
import os
import pickle
import random
import re
import time
from collections import Counter
from operator import mul

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEC_PATH = os.path.join(BASE_DIR, "growth", "char_vecs.pkl")
CORPORA = (os.path.join(BASE_DIR, "corpus_large.tsv"),
           os.path.join(BASE_DIR, "corpus.txt"))

SEM_VER = 3          # 算法版本（变即重训）
DIM = 32             # 向量维度
WINDOW = 4           # 上下文窗口（4：捕捉话题级共现，压过相邻字的语法性共现）
NEG = 3              # 每个正样本配的负样本数
MIN_COUNT = 10       # 字频下限（进词表；低频字训练不足只会带来噪声）
TRAIN_P = 0.35       # 中心字采样率（训练预算控制）
LR0, LR_MIN = 0.025, 1e-4
SUB_T = 1e-4         # 高频字下采样阈值（的/是 之类少训不损语义）
TABLE_SIZE = 60000   # 负采样表大小
SEED = 20260822

# 单字功能词：句子向量里剔除（与 2-gram 停用表同理，匹配比实义字）
STOPCHARS = frozenset(
    "的了呢吧啊呀哦嘛么是有在和与或就也都还又再很太不没我你他她它这那个些"
    "怎么怎样如何为什多少是否只被把给让从对跟并而且但如果所以虽然然后"
)

_NORM_RE = re.compile(r"[\s，。？！,?!：:、.·\-~\"'（）()\[\]【】]+")


def norm(s):
    return _NORM_RE.sub("", s.lower())


def _sigmoid(s):
    if s > 20.0:
        return 1.0
    if s < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-s))


class SemanticSpace:
    """字符语义向量空间。句子向量 = 成员字向量的 IDF 加权平均（L2 归一）。"""

    def __init__(self, vectors, dim):
        self.vectors = vectors      # {char: [float]*dim}
        self.dim = dim

    def vec(self, text, idf=None, idf_default=None):
        """文本 → 归一化语义向量。没有可用实义字返回 None。"""
        chars = [c for c in norm(text)
                 if c in self.vectors and c not in STOPCHARS]
        if not chars:
            return None
        acc = [0.0] * self.dim
        if idf:
            for c in chars:
                u = self.vectors[c]
                w = idf.get(c, idf_default or 1.0)
                acc = [a + w * b for a, b in zip(acc, u)]
        else:
            for c in chars:
                u = self.vectors[c]
                acc = [a + b for a, b in zip(acc, u)]
        n = math.sqrt(sum(x * x for x in acc)) or 1.0
        return [x / n for x in acc]

    @staticmethod
    def cos(v1, v2):
        return sum(map(mul, v1, v2))

    # ------------------------------------------------------------------
    # 加载 / 训练（自带缓存）
    # ------------------------------------------------------------------

    @classmethod
    def load_or_train(cls, verbose=True):
        key = space_fingerprint()
        if os.path.exists(VEC_PATH):
            try:
                with open(VEC_PATH, "rb") as fh:
                    d = pickle.load(fh)
                if d.get("key") == key:
                    if verbose:
                        print(f"  语义空间：缓存命中 "
                              f"({len(d['vectors'])} 字 / {d['dim']} 维)")
                    return cls(d["vectors"], d["dim"])
            except Exception:
                pass
        if verbose:
            print("  语义空间：从语料训练字符向量（首次约 1-2 分钟）…")
        t0 = time.time()
        vectors = _train(verbose)
        # 中心化 + 逐字归一：去掉全体向量的公共偏置方向，
        # 否则任何两句的余弦都 ~0.95（全部挤在同一方向，分辨不出远近）
        n_v = len(vectors)
        mean = [sum(v[i] for v in vectors.values()) / n_v for i in range(DIM)]
        vectors = {c: _unit([x - m for x, m in zip(v, mean)])
                   for c, v in vectors.items()}
        os.makedirs(os.path.dirname(VEC_PATH), exist_ok=True)
        tmp = VEC_PATH + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"key": key, "dim": DIM, "vectors": vectors,
                         "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                        fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, VEC_PATH)
        if verbose:
            print(f"  语义空间训练完成：{len(vectors)} 字 × {DIM} 维，"
                  f"用时 {time.time() - t0:.0f}s")
        return cls(vectors, DIM)


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def space_fingerprint():
    fp = [("sem_ver", SEM_VER), ("params", (DIM, WINDOW, NEG, TRAIN_P)),
          ("centered", 1)]
    for p in CORPORA:
        if os.path.exists(p):
            st = os.stat(p)
            fp.append((p, st.st_size, st.st_mtime))
    return fp


# ----------------------------------------------------------------------
# 训练器：skip-gram + 负采样
# ----------------------------------------------------------------------

def _train(verbose=True):
    rng = random.Random(SEED)

    # 1. 语料流 + 字频统计
    lines = []
    freq = Counter()
    for path in CORPORA:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                text = "".join(raw.split("\t")[:2])
                if len(text) < 4:
                    continue
                lines.append(text)
                freq.update(text)
    total = sum(freq.values())
    vocab = {c for c, n in freq.items() if n >= MIN_COUNT}
    if verbose:
        print(f"    语料 {len(lines)} 行 / {total} 字，词表 {len(vocab)} 字")

    # 2. 高频字下采样概率（的/是/我 这类少训，省预算不损语义）
    keep = {}
    for c in vocab:
        f = freq[c] / total
        keep[c] = 1.0 if f <= SUB_T else min(1.0, math.sqrt(SUB_T / f) + SUB_T / f)

    # 3. 负采样表（按词频 0.75 次方）
    pow_sum = sum(n ** 0.75 for c, n in freq.items() if c in vocab)
    neg_table = []
    for c, n in freq.items():
        if c in vocab:
            rep = max(1, int(TABLE_SIZE * (n ** 0.75) / pow_sum))
            neg_table.extend([c] * rep)
    table_len = len(neg_table)

    # 4. 双矩阵初始化（中心向量 + 上下文向量）
    vec = {c: [rng.uniform(-0.5, 0.5) / DIM for _ in range(DIM)] for c in vocab}
    ctx = {c: [0.0] * DIM for c in vocab}

    # 5. 学习率调度（按预计更新量线性衰减）
    est_kept = sum(freq[c] * keep[c] for c in vocab)
    est_centers = est_kept * TRAIN_P
    est_updates = max(1, int(est_centers * 2 * WINDOW * 0.75 * (1 + NEG)))
    done = 0
    next_report = est_updates // 10

    # 6. 主循环
    t0 = time.time()
    randrange = rng.randrange
    random_ = rng.random
    for text in lines:
        pos = [c for c in text if c in vocab and random_() < keep[c]]
        L = len(pos)
        if L < 2:
            continue
        for i in range(L):
            if random_() > TRAIN_P:
                continue
            c = pos[i]
            vc = vec[c]
            lo = i - WINDOW if i > WINDOW else 0
            hi = i + WINDOW + 1
            if hi > L:
                hi = L
            for j in range(lo, hi):
                if j == i:
                    continue
                lr = LR0 * (1.0 - done / est_updates)
                if lr < LR_MIN:
                    lr = LR_MIN
                o = pos[j]
                # —— 正样本：c 的上下文里真的出现过 o ——
                uo = ctx[o]
                g = (1.0 - _sigmoid(sum(map(mul, vc, uo)))) * lr
                if g:
                    vc[:] = [x + g * y for x, y in zip(vc, uo)]
                    uo[:] = [y + g * x for x, y in zip(vc, uo)]
                # —— 负样本：随机字，大概率无关 ——
                for _ in range(NEG):
                    neg_c = neg_table[randrange(table_len)]
                    if neg_c == c:
                        continue
                    un = ctx[neg_c]
                    g = -_sigmoid(sum(map(mul, vc, un))) * lr
                    if g:
                        vc[:] = [x + g * y for x, y in zip(vc, un)]
                        un[:] = [y + g * x for x, y in zip(vc, un)]
                done += 1 + NEG
                if done >= next_report:
                    if verbose:
                        pct = min(100, done * 100 // est_updates)
                        print(f"    … {pct}%（{time.time() - t0:.0f}s）")
                    next_report += est_updates // 10
    return vec


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    sp = SemanticSpace.load_or_train(verbose=True)
    # IDF：按问题列字频算（模拟真实检索时的加权）
    from collections import Counter
    df = Counter()
    N = 0
    for path in CORPORA:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                parts = raw.split("\t")
                if len(parts) >= 2:
                    N += 1
                    df.update({c for c in norm(parts[0]) if c not in STOPCHARS})
    idf = {c: math.log(N / (n + 1.0)) + 0.5 for c, n in df.items()}
    default = math.log(N) + 0.5
    tests = [("背单词记得牢", "怎样提高记忆力"),
             ("背单词记得牢", "红烧肉怎么做"),
             ("黑洞", "天体"),
             ("黑洞", "简历"),
             ("怎么拒绝别人", "如何婉拒他人"),
             ("怎么拒绝别人", "感冒了怎么办")]
    print("\n语义距离自检（IDF 加权 cosine，越高越近）：")
    for a, b in tests:
        va = sp.vec(a, idf, default)
        vb = sp.vec(b, idf, default)
        if va and vb:
            print(f"  {a} ↔ {b}: {SemanticSpace.cos(va, vb):.3f}")
        else:
            print(f"  {a} ↔ {b}: （字不在词表）")
