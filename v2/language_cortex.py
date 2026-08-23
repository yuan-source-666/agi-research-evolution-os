# -*- coding: utf-8 -*-
"""
SEED OS v2.2 — language_cortex.py
语言皮层：联想记忆器官（大规模版）。

设计原则（与主架构一致——结构即智能）：
- 无参数回归、无向量嵌入。联想 = 字符 2-gram 模式的共现键。
- 倒排索引：问题与答案首句双通道 gram → 记忆编号。检索只看候选，
  十万级记忆毫秒级联想；话题词只出现在答案里也能被追问召回。
- 赫布律：重复学习强化键（weight ↑），同场竞争者轻微衰减（遗忘）。
- 回答分级（v2.5）：确定带断言 / 猜测带有依据地猜 / 联想带交出沾边的记忆 /
  边缘带坦白摸到什么——人不因没背过原文就闭嘴，只有零重叠才认输。

数据源：
- corpus.txt            小种子核心语料（身份/物理知识），高权重
- corpus_large.tsv      大规模语料（Belle 1M 过滤版），海量常识
- growth/learned_pairs.jsonl  老板随教随存的（最高优先级）
- growth/cortex_cache.pkl     训练结果缓存，重启秒加载
"""
import json
import math
import os
import pickle
import re
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNED_PATH = os.path.join(BASE_DIR, "growth", "learned_pairs.jsonl")
LARGE_PATH = os.path.join(BASE_DIR, "corpus_large.tsv")
CACHE_PATH = os.path.join(BASE_DIR, "growth", "cortex_cache.pkl")

W_MAX = 5.0        # 键强上限（防单条记忆过强垄断）
W_LEARN = 1.0      # 每次学习的增量
W_DECAY = 0.002    # 同场竞争失败者的衰减（遗忘）
T_SURE = 0.42      # 确定带：直接断言
T_GUESS = 0.24     # 猜测带：有依据地说"我记得大概是"
T_ASSOC = 0.12     # 联想带：交出沾边的记忆，标注不确定
CAND_TOP = 400     # 每次联想参与精算的候选数上限
# 跨词边界伪词过滤器：以虚字开头/结尾的 2-gram 多为切片碎片（『在高』『原上』），
# 不是真词。追问解析话题种子时先过这道筛。
FUNC_HEAD = frozenset("在那这个是有和与或很被把从向对就都已经还才刚只")
FUNC_TAIL = frozenset("的了呢吗吧啊上下中里去来是要会是能有和与地得着过们")
CORTEX_VER = 4     # 皮层算法版本（影响缓存指纹）
GUESS_SAY = ("我记得大概是这么回事：", "如果我没记错的话：", "印象里应该是：")

# 高频功能词 2-gram：在联想匹配里是纯噪音（"什么/怎么/如何"谁都共享），
# 训练和查询两侧同步剔除，让匹配真正比的是实义词。
STOPGRAMS = frozenset({
    "什么", "怎么", "怎样", "如何", "为什么", "可以", "我们", "你们", "他们",
    "还是", "的话", "就是", "现在", "自己", "这个", "那个", "一个", "有些",
    "哪些", "有没有", "是不是", "能不能", "应该", "可能", "觉得", "知道",
    "时候", "地方", "东西", "问题", "情况", "方面", "进行", "出现", "关于",
    "对于", "以及", "或者", "但是", "而且", "然后", "所以", "如果", "虽然",
    "多少", "几个", "一些", "这些", "那些", "这样", "那样", "非常", "真的",
    "请问", "一下", "我想", "我要", "帮我", "给我", "有没有",
})


def _norm(s):
    return re.sub(r"[\s，。？！,?!：:、.·\-~\"'（）()\[\]【】]+", "", s.lower())


def _grams(s):
    s = _norm(s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)} - STOPGRAMS


class LanguageCortex:
    """联想记忆：问题模式 → 回答，倒排索引 + 共现键强度。"""

    def __init__(self):
        self.memories = []     # [{"q":str, "a":str, "w":float}]
        self._norm2idx = {}    # 归一化问题 → 记忆编号（去重 + 精确命中）
        self._index = {}       # 2-gram → [记忆编号]（问题倒排索引）
        self._aindex = {}      # 2-gram → [记忆编号]（答案首句倒排索引，v2.6）
        self.hits = 0
        self.misses = 0
        self.mode_counts = {"sure": 0, "guess": 0, "associate": 0,
                            "edge": 0, "clueless": 0}

    # ------------------------------------------------------------------
    # 学习（赫布强化）
    # ------------------------------------------------------------------

    def learn(self, q, a, times=1):
        """学一对问答。相同问题重复学习 → 键变强（赫布律）。"""
        q, a = q.strip(), a.strip()
        if not q or not a or "\n" in q or "\n" in a or "\t" in q or "\t" in a:
            return False
        nq = _norm(q)
        if not nq:
            return False
        idx = self._norm2idx.get(nq)
        if idx is not None:                      # 重复学习 → 强化
            m = self.memories[idx]
            m["a"] = a
            m["w"] = min(W_MAX, m["w"] + W_LEARN * times)
            return True
        self.memories.append({"q": q, "a": a, "w": W_LEARN * times})
        i = len(self.memories) - 1
        self._norm2idx[nq] = i
        for g in _grams(q):
            self._index.setdefault(g, []).append(i)
        for g in _grams(self._first_sentence(a)):
            self._aindex.setdefault(g, []).append(i)
        return True

    def train_from_corpus(self, path, epochs=1, times=1, verbose=False):
        """从语料库训练（早读：重复朗读强化联想）。TSV 格式：问题<TAB>回答。"""
        if not os.path.exists(path):
            return 0
        n = 0
        for _ in range(epochs):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2 and self.learn(parts[0], parts[1], times):
                        n += 1
        if verbose:
            print(f"  语料 {os.path.basename(path)}：本库现 {len(self.memories)} 组记忆")
        return n

    def train_from_learned(self, path=LEARNED_PATH):
        """加载老板交互教学中存的问答（学过的 = 长期记忆，最高优先）。
        权重 ×5：老板亲口教的压过语料库的泛泛之谈。"""
        if not os.path.exists(path):
            return 0
        n = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    if self.learn(d.get("q", ""), d.get("a", ""), times=5):
                        n += 1
                except json.JSONDecodeError:
                    continue
        return n

    # ------------------------------------------------------------------
    # 大规模训练 + 缓存
    # ------------------------------------------------------------------

    def train_all(self, verbose=True):
        """标准启动流程：核心语料 ×3 遍 + 大语料 + 老板教的话。
        大语料训练结果有 pickle 缓存，重启秒加载。"""
        t0 = time.time()
        key = self._fingerprint()
        if os.path.exists(CACHE_PATH) and self._cache_valid(key):
            if verbose:
                print("  语言皮层：读取训练缓存 …")
            with open(CACHE_PATH, "rb") as fh:
                data = pickle.load(fh)
            self.memories = data["memories"]
            self._norm2idx = data["norm2idx"]
            self._index = data["index"]
            self._aindex = data.get("aindex", {})
            self.train_from_learned()
            if verbose:
                print(f"  缓存命中：{len(self.memories)} 组记忆 "
                      f"({time.time() - t0:.1f}s)")
            return

        if verbose:
            print("  语言皮层：核心语料早读 3 遍 …")
        self.train_from_corpus(os.path.join(BASE_DIR, "corpus.txt"),
                               epochs=3, times=1)
        if os.path.exists(LARGE_PATH):
            if verbose:
                sz = os.path.getsize(LARGE_PATH) / 1e6
                print(f"  语言皮层：消化大语料 corpus_large.tsv（{sz:.0f} MB）…")
            self.train_from_corpus(LARGE_PATH, epochs=1, times=1)
        n_learned = self.train_from_learned()
        if verbose:
            print(f"  语言皮层训练完成：{len(self.memories)} 组记忆 "
                  f"+ 老板教过的 {n_learned} 组，用时 {time.time() - t0:.1f}s")
        self._save_cache(key)

    def _fingerprint(self):
        fp = [("cortex_ver", CORTEX_VER)]   # 皮层算法版本（变索引结构即重训）
        for p in (os.path.join(BASE_DIR, "corpus.txt"), LARGE_PATH):
            if os.path.exists(p):
                st = os.stat(p)
                fp.append((p, st.st_size, st.st_mtime))
        return fp

    def _cache_valid(self, key):
        try:
            with open(CACHE_PATH, "rb") as fh:
                return pickle.load(fh).get("key") == key
        except Exception:
            return False

    def _save_cache(self, key):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"key": key, "memories": self.memories,
                         "norm2idx": self._norm2idx, "index": self._index,
                         "aindex": self._aindex},
                        fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, CACHE_PATH)

    # ------------------------------------------------------------------
    # 联想（倒排索引检索 + 竞争遗忘）
    # ------------------------------------------------------------------

    def _first_sentence(self, a, limit=72):
        """截取回答的第一句：联想带只交出最有把握的那一段，不硬撑整段背诵。"""
        for ch in ("。", "！", "？", "；"):
            p = a.find(ch)
            if 0 < p <= limit:
                return a[:p + 1]
        return a[:limit] + ("…" if len(a) > limit else "")

    def _idf(self, gram):
        """逆文档频率：越稀有的字词越能代表一个问题。
        df 就是倒排表长度——零额外存储。"""
        return math.log(1.0 + len(self.memories)
                        / (1 + len(self._index.get(gram, ()))))

    def respond(self, text, prev=None):
        """分级联想应答入口（v2.6）。prev=上一轮话题：
        短追问（『那声音呢』『它呢』）会借上一话题一起联想——
        像人接着话头说话，而不是每次都当新句子。借不上力再按原问。"""
        raw = text.strip()
        if prev:
            nq, npv = _norm(text), _norm(prev)
            short = (len(nq) <= 4 or bool(re.match(r"^[那这它他她]", nq))
                     or nq.endswith("呢") or nq.endswith("勒"))
            if short and npv and npv != nq:
                h0, ms0, mc0 = self.hits, self.misses, dict(self.mode_counts)
                # 话头接续：只按新内容词块召回（『那声音呢』→ 声音），旧问句当疑问框架。
                # 语境本身就是证据——短追问允许更低的相似度门槛。
                g_new = [x for x in _grams(text) if x not in set(_grams(prev))]
                r = self._respond_fresh(g_new, npv, raw) if g_new else None
                if r and r.get("mode") in ("sure", "guess", "associate"):
                    r["via_context"] = True
                    return r
                # 借不上力 → 回滚计数，按原问题独立应答
                self.hits, self.misses, self.mode_counts = h0, ms0, mc0
        return self._respond(raw)

    def _respond_fresh(self, fresh_grams, frame_norm, raw):
        """短追问续接（v2.6）：解析出追问的话题种子（df≥3 的实义词块，
        问题与答案首句双通道召回），像人接着话头换主语再问一次。
        问式与上一问句接近 → 直接断言；否则按猜测口吻交付。
        没有任何成气候的话题种子 → None（交给独立应答）。"""
        cand_grams = {g for g in fresh_grams
                      if len(self._index.get(g, ()))
                      + len(self._aindex.get(g, ())) >= 3
                      and g[0] not in FUNC_HEAD and g[-1] not in FUNC_TAIL}
        # 共字遮蔽：与更高频候选共享汉字的碎片（『在高』之于『高原』）
        # 多是跨词边界的伪词，不配当话题。
        ordered = sorted(cand_grams,
                         key=lambda g: -(len(self._index.get(g, ()))
                                         + len(self._aindex.get(g, ()))))
        seeds = []
        for g in ordered:
            if not any(set(g) & set(k) for k in seeds):
                seeds.append(g)
            if len(seeds) >= 3:
                break
        if not seeds:
            return None
        cand = {}
        for gram in seeds:
            w = self._idf(gram)
            for i in self._index.get(gram, ()):       # 问题侧命中
                cand[i] = cand.get(i, 0.0) + w
            for i in self._aindex.get(gram, ()):      # 答案侧命中（同权）
                cand[i] = cand.get(i, 0.0) + w
        if not cand:
            return None
        seed_w = sum(self._idf(x) for x in seeds) or 1.0
        fg = set(_grams(frame_norm))
        ranked = []
        for i, sw in sorted(cand.items(), key=lambda kv: -kv[1])[:CAND_TOP]:
            m = self.memories[i]
            me = _grams(m["q"]) | _grams(self._first_sentence(m["a"], limit=60))
            coverage = min(1.0, sw / seed_w)                # 种子覆盖度（门槛）
            frame_ratio = len(me & fg) / max(1, len(fg))    # 问式重合度（意图）
            # 语境模式下键强只做 ±7% 微调：语料里的高频幸运儿不该压过语境证据
            conf = 0.85 + 0.15 * min(1.0, m["w"] / W_MAX)
            # 乘性组合：话题不对=一票否决；话题对时靠问式挑出正确意图
            score = math.sqrt(coverage) * (0.35 + 0.65 * frame_ratio) * conf
            ranked.append((i, m, score, coverage, frame_ratio))
        ranked.sort(key=lambda t: (-t[2], -t[4], -t[0]))
        best_i, best, best_score, coverage, frame_ratio = ranked[0]
        near = [m["q"][:20] for _, m, _, _, _ in ranked[1:3]]
        best["w"] = min(W_MAX, best["w"] + 0.05)   # 越用越牢
        mode = "sure" if frame_ratio >= 0.30 else "guess"
        ans = best["a"]
        if mode == "guess":
            say = GUESS_SAY[len(best["q"]) % len(GUESS_SAY)]
            if len(ans) > 90:               # 不确信就不背诵长文
                ans = self._first_sentence(ans, limit=88)
            ans = f"{say}{ans}"
        self.hits += 1
        self.mode_counts[mode] += 1
        return {"answer": ans, "mode": mode,
                "similarity": round(min(1.0, coverage), 3),
                "source": best["q"],
                "strength": round(min(1.0, best["w"] / W_MAX), 2),
                "seeds": seeds}

    def _respond(self, text):
        """分级联想应答（v2.5）。永远给出回应：
        sure 断言 / guess 有依据地猜 / associate 交出沾边的记忆 /
        edge 坦白只摸到零星字词 / clueless 全无重叠才真正认输。"""
        raw = text.strip()
        if not self.memories:
            self.misses += 1
            self.mode_counts["clueless"] += 1
            return {"answer": "我脑子里还是空的，什么都还没学。", "mode": "clueless",
                    "similarity": 0.0}

        nq = _norm(text)
        # —— 精确命中（问过/教过的原话）直接激活 ——
        idx = self._norm2idx.get(nq)
        if idx is not None:
            m = self.memories[idx]
            m["w"] = min(W_MAX, m["w"] + 0.05)
            self.hits += 1
            self.mode_counts["sure"] += 1
            return {"answer": m["a"], "mode": "sure",
                    "strength": round(min(1.0, m["w"] / W_MAX), 2),
                    "similarity": 1.0, "source": m["q"]}

        g = _grams(text)
        if not g:
            self.misses += 1
            self.mode_counts["clueless"] += 1
            return {"answer": ("这串字符我读不出任何认识的词。要让我记住它，"
                               f"可以教我：『学习 {raw}|你的解释』"),
                    "mode": "clueless", "similarity": 0.0}

        # —— IDF 加权召回：稀有字词话语权大，功能词残留干扰小 ——
        cand = {}                      # 记忆编号 -> 共享 gram 的 idf 累计
        cand_n = {}                    # 记忆编号 -> 共享 gram 个数（分级门槛用）
        for gram in g:
            w = self._idf(gram)
            for i in self._index.get(gram, ()):
                cand[i] = cand.get(i, 0.0) + w
                cand_n[i] = cand_n.get(i, 0) + 1
        if not cand:
            self.misses += 1
            self.mode_counts["clueless"] += 1
            return {"answer": (f"这个真把我难住了——{len(self.memories)} 组记忆里"
                               f"没有一条沾边。你教我一次我就永久记住："
                               f"『学习 {raw}|答案』"),
                    "mode": "clueless", "similarity": 0.0}

        qw = sum(self._idf(x) for x in g)
        ranked = []
        for i, sw in sorted(cand.items(), key=lambda kv: -kv[1])[:CAND_TOP]:
            m = self.memories[i]
            mg = _grams(m["q"])
            mw = sum(self._idf(x) for x in mg) or 1.0
            sim = 2.0 * sw / (qw + mw)          # IDF 加权 Dice
            conf = min(1.0, 0.55 + 0.09 * m["w"])   # 键强→信心（连续，不提前封顶）
            ranked.append((i, m, sim * conf, sim))
        ranked.sort(key=lambda t: (-t[2], -t[0]))   # 平局时新近学会的优先
        # 赫布遗忘：竞争失败者轻微衰减（新陈代谢）
        for _, m, _, _ in ranked[1:201]:
            m["w"] = max(0.5, m["w"] - W_DECAY)

        best_i, best, best_score, best_sim = ranked[0]
        near = [m["q"][:20] for _, m, _, _ in ranked[1:3]]
        return self._answer_tier(best, best_score, best_sim, raw,
                                 shared=cand_n.get(best_i, 0), near=near)

    def _answer_tier(self, best, best_score, best_sim, raw, shared, near=None,
                     t_sure=T_SURE, t_guess=T_GUESS, min_shared=2):
        """分级应答决策（共用）：确定带断言 / 猜测带有依据地猜 /
        联想带交出沾边的记忆 / 边缘带坦白只摸到零星词。
        语境续接时阈值可放宽（t_sure/t_guess/min_shared）——语境即证据。"""
        near = near or []
        best["w"] = min(W_MAX, best["w"] + 0.05)   # 越用越牢
        src = best["q"]
        meta = {"similarity": round(best_sim, 3), "source": src,
                "strength": round(min(1.0, best["w"] / W_MAX), 2)}

        # —— 分级应答 ——
        if best_score >= t_sure:
            self.hits += 1
            self.mode_counts["sure"] += 1
            return {"answer": best["a"], "mode": "sure", **meta}
        if best_score >= t_guess:
            self.hits += 1
            self.mode_counts["guess"] += 1
            say = GUESS_SAY[len(src) % len(GUESS_SAY)]
            ans = best["a"]
            if len(ans) > 90:               # 不确信就不背诵长文
                ans = self._first_sentence(ans, limit=88)
            return {"answer": f"{say}{ans}", "mode": "guess", **meta}
        if best_score >= T_ASSOC and shared >= max(min_shared, 4):
            self.hits += 1
            self.mode_counts["associate"] += 1
            return {"answer": (f"这个我没正经学过，不过它让我想起学过的『{src}』——"
                               f"{self._first_sentence(best['a'])}"
                               f"要是答偏了，你教我：『学习 {raw}|答案』"),
                    "mode": "associate", **meta}
        # —— 边缘带：只有零星字词沾边，坦白 + 给出可追问的方向 ——
        self.misses += 1
        if shared < min_shared:         # 词块重合低于底线——真不认识
            self.mode_counts["clueless"] += 1
            return {"answer": (f"这个真把我难住了——{len(self.memories)} 组记忆里"
                               f"没有一条真正沾边。你教我一次我就永久记住："
                               f"『学习 {raw}|答案』"),
                    "mode": "clueless", **meta}
        self.mode_counts["edge"] += 1
        listed = f"『{src[:20]}』" + "".join(f"和『{x}』" for x in near)
        return {"answer": (f"这个问题我只摸到点边——记忆里最接近的是{listed}。"
                           f"换个问法我可能就想起来了；或者直接教我："
                           f"『学习 {raw}|答案』"),
                "mode": "edge", **meta}

    # ------------------------------------------------------------------
    # 持久化（老板教的话永不丢）
    # ------------------------------------------------------------------

    def save_learned(self, q, a, path=LEARNED_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"q": q, "a": a, "at": time.strftime("%Y-%m-%d %H:%M:%S")},
                                ensure_ascii=False) + "\n")

    def stats(self):
        return {"memories": len(self.memories),
                "grams": len(self._index),
                "hits": self.hits, "misses": self.misses,
                "modes": dict(self.mode_counts)}
