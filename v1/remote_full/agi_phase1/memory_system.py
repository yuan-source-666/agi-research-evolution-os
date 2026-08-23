"""
AGI Memory System v0.1
======================
分层记忆架构：工作记忆 → 情景记忆 → 语义记忆 → 程序记忆
支持：存储、检索、固化（episodic→semantic）、压缩、遗忘、冲突解决

依赖：仅标准库 + sqlite3（Python 内置）
设计原则：
  - 无外部依赖，可在任何环境运行
  - 所有知识保留来源、时间、可信度、验证状态
  - 支持主动 consolidation（情景→语义）
  - 支持基于重要性的遗忘
  - 线程安全

用法：
  from memory_system import MemorySystem
  mem = MemorySystem("./memory_store")
  mem.store_episodic("用户问了AGI架构问题", source="conversation", importance=0.8)
  results = mem.retrieve("AGI架构", top_k=5)
  mem.consolidate()  # 情景→语义固化
"""

import json
import sqlite3
import time
import hashlib
import os
import threading
from typing import Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class MemoryType(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class EvidenceLevel(Enum):
    FACT = "FACT"
    HYPOTHESIS = "HYPOTHESIS"
    UNKNOWN = "UNKNOWN"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class MemoryItem:
    """一条记忆"""
    id: str = ""
    type: str = "episodic"          # working/episodic/semantic/procedural
    content: str = ""               # 记忆内容
    source: str = ""                # 来源
    timestamp: float = 0.0           # 创建时间
    last_access: float = 0.0        # 最后访问时间
    access_count: int = 0           # 访问次数
    importance: float = 0.5          # 重要性 0-1
    confidence: float = 0.5         # 可信度 0-1
    evidence: str = "UNKNOWN"       # 证据等级
    tags: list = field(default_factory=list)  # 标签
    embedding: list = field(default_factory=list)  # 简易词频向量（无外部依赖）
    consolidated_from: str = ""     # 固化来源（情景记忆ID列表）
    version: int = 1                # 版本号
    expired: bool = False           # 是否过期

    def to_dict(self):
        d = asdict(self)
        d["tags"] = json.dumps(self.tags, ensure_ascii=False)
        d["embedding"] = json.dumps(self.embedding)
        return d

    @classmethod
    def from_row(cls, row):
        """从数据库行构造"""
        item = cls()
        item.id = row["id"]
        item.type = row["type"]
        item.content = row["content"]
        item.source = row["source"]
        item.timestamp = row["timestamp"]
        item.last_access = row["last_access"]
        item.access_count = row["access_count"]
        item.importance = row["importance"]
        item.confidence = row["confidence"]
        item.evidence = row["evidence"]
        item.tags = json.loads(row["tags"]) if row["tags"] else []
        item.embedding = json.loads(row["embedding"]) if row["embedding"] else []
        item.consolidated_from = row["consolidated_from"] or ""
        item.version = row["version"]
        item.expired = bool(row["expired"])
        return item


class SimpleEmbedding:
    """简易词频向量——无外部依赖的文本向量化
    用 TF（词频）做向量，cosine 相似度做检索。
    够用且可测试。后续可替换为真实 embedding。
    """

    def __init__(self):
        self.vocab = {}  # word -> index
        self.lock = threading.Lock()

    def _tokenize(self, text):
        """简易分词：英文按词，中文按字 + bigram
        加入中文 bigram 提高短文本相似度匹配率。
        """
        import re
        # 英文单词
        tokens = re.findall(r'[a-zA-Z]+', text.lower())
        # 中文字符（单字）
        cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens += cn_chars
        # 中文 bigram（相邻两字组成词），提高语义匹配
        for i in range(len(cn_chars) - 1):
            tokens.append(cn_chars[i] + cn_chars[i + 1])
        return tokens

    def embed(self, text):
        tokens = self._tokenize(text)
        vec = {}
        for t in tokens:
            with self.lock:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
            idx = self.vocab[t]
            vec[idx] = vec.get(idx, 0) + 1
        # 转为稀疏列表（只存非零）
        return [[k, v] for k, v in sorted(vec.items())]

    @staticmethod
    def cosine_sim(v1, v2):
        """稀疏向量 cosine 相似度"""
        d1 = dict(v1)
        d2 = dict(v2)
        all_keys = set(d1.keys()) | set(d2.keys())
        dot = sum(d1.get(k, 0) * d2.get(k, 0) for k in all_keys)
        n1 = sum(v * v for v in d1.values()) ** 0.5
        n2 = sum(v * v for v in d2.values()) ** 0.5
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)


class MemorySystem:
    """AGI 分层记忆系统"""

    def __init__(self, store_path: str):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self.db_path = os.path.join(store_path, "memory.db")
        self.embedder = SimpleEmbedding()
        self.lock = threading.RLock()

        # 工作记忆（内存中，不持久化）
        self._working: list[MemoryItem] = []
        self._working_max = 20  # 工作记忆容量

        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT DEFAULT '',
                    timestamp REAL DEFAULT 0,
                    last_access REAL DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 0.5,
                    evidence TEXT DEFAULT 'UNKNOWN',
                    tags TEXT DEFAULT '[]',
                    embedding TEXT DEFAULT '[]',
                    consolidated_from TEXT DEFAULT '',
                    version INTEGER DEFAULT 1,
                    expired INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
            conn.commit()
            conn.close()

    def _gen_id(self, content, ts):
        raw = f"{content[:100]}{ts}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    # ========== 存储 ==========

    def store_working(self, content, importance=0.5, tags=None):
        """存入工作记忆（内存）"""
        with self.lock:
            item = MemoryItem(
                id=self._gen_id(content, time.time()),
                type=MemoryType.WORKING.value,
                content=content,
                timestamp=time.time(),
                last_access=time.time(),
                importance=importance,
                tags=tags or []
            )
            self._working.append(item)
            # 超容量时淘汰最不重要的
            if len(self._working) > self._working_max:
                self._working.sort(key=lambda x: x.importance * x.access_count)
                evicted = self._working.pop(0)
                # 重要的工作记忆转入情景记忆
                if evicted.importance > 0.5:
                    self.store_episodic(evicted.content, source="working_memory_overflow",
                                       importance=evicted.importance, tags=evicted.tags)
            return item.id

    def store_episodic(self, content, source="", importance=0.5, tags=None,
                       confidence=0.5, evidence="UNKNOWN"):
        """存入情景记忆（持久化）"""
        with self.lock:
            ts = time.time()
            item = MemoryItem(
                id=self._gen_id(content, ts),
                type=MemoryType.EPISODIC.value,
                content=content,
                source=source,
                timestamp=ts,
                last_access=ts,
                importance=importance,
                confidence=confidence,
                evidence=evidence,
                tags=tags or [],
                embedding=self.embedder.embed(content)
            )
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, type, content, source, timestamp, last_access, access_count,
                 importance, confidence, evidence, tags, embedding, consolidated_from, version, expired)
                VALUES (:id, :type, :content, :source, :timestamp, :last_access, :access_count,
                        :importance, :confidence, :evidence, :tags, :embedding, :consolidated_from, :version, :expired)
            """, item.to_dict())
            conn.commit()
            conn.close()
            return item.id

    def store_semantic(self, content, source="", importance=0.7, tags=None,
                       confidence=0.5, evidence="HYPOTHESIS", consolidated_from=""):
        """存入语义记忆（持久化，通常是固化的结果）"""
        with self.lock:
            ts = time.time()
            item = MemoryItem(
                id=self._gen_id(content, ts),
                type=MemoryType.SEMANTIC.value,
                content=content,
                source=source,
                timestamp=ts,
                last_access=ts,
                importance=importance,
                confidence=confidence,
                evidence=evidence,
                tags=tags or [],
                embedding=self.embedder.embed(content),
                consolidated_from=consolidated_from
            )
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, type, content, source, timestamp, last_access, access_count,
                 importance, confidence, evidence, tags, embedding, consolidated_from, version, expired)
                VALUES (:id, :type, :content, :source, :timestamp, :last_access, :access_count,
                        :importance, :confidence, :evidence, :tags, :embedding, :consolidated_from, :version, :expired)
            """, item.to_dict())
            conn.commit()
            conn.close()
            return item.id

    def store_procedural(self, content, source="", importance=0.7, tags=None,
                         confidence=0.5, evidence="HYPOTHESIS"):
        """存入程序记忆（技能/方法）"""
        with self.lock:
            ts = time.time()
            item = MemoryItem(
                id=self._gen_id(content, ts),
                type=MemoryType.PROCEDURAL.value,
                content=content,
                source=source,
                timestamp=ts,
                last_access=ts,
                importance=importance,
                confidence=confidence,
                evidence=evidence,
                tags=tags or [],
                embedding=self.embedder.embed(content)
            )
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, type, content, source, timestamp, last_access, access_count,
                 importance, confidence, evidence, tags, embedding, consolidated_from, version, expired)
                VALUES (:id, :type, :content, :source, :timestamp, :last_access, :access_count,
                        :importance, :confidence, :evidence, :tags, :embedding, :consolidated_from, :version, :expired)
            """, item.to_dict())
            conn.commit()
            conn.close()
            return item.id

    # ========== 检索 ==========

    def retrieve(self, query, top_k=5, memory_types=None, min_importance=0.0):
        """语义检索：从持久化记忆中检索最相关的 top_k 条"""
        with self.lock:
            query_vec = self.embedder.embed(query)
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            types_filter = ""
            params = []
            if memory_types:
                placeholders = ",".join("?" * len(memory_types))
                types_filter = f"AND type IN ({placeholders})"
                params = list(memory_types)
            params.extend([min_importance])

            rows = conn.execute(f"""
                SELECT * FROM memories
                WHERE expired = 0 AND importance >= ?
                {types_filter}
                ORDER BY importance DESC, timestamp DESC
                LIMIT 200
            """, params).fetchall()

            results = []
            for row in rows:
                item = MemoryItem.from_row(row)
                sim = SimpleEmbedding.cosine_sim(query_vec, item.embedding)
                # 综合分数 = 语义相似度 * 重要性 * 时间衰减
                age = time.time() - item.timestamp
                time_decay = 1.0 / (1.0 + age / 86400)  # 按天衰减
                score = sim * item.importance * (0.5 + 0.5 * time_decay)
                results.append((score, item))

            results.sort(key=lambda x: x[0], reverse=True)
            results = results[:top_k]

            # 更新访问记录
            for score, item in results:
                conn.execute("""
                    UPDATE memories SET last_access = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (time.time(), item.id))
            conn.commit()
            conn.close()

            return [(score, item) for score, item in results]

    def get_working(self):
        """获取工作记忆"""
        with self.lock:
            return list(self._working)

    def get_all(self, memory_type=None, limit=100):
        """获取指定类型的所有记忆"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            if memory_type:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE type = ? AND expired = 0 ORDER BY timestamp DESC LIMIT ?",
                    (memory_type, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE expired = 0 ORDER BY timestamp DESC LIMIT ?",
                    (limit,)).fetchall()
            conn.close()
            return [MemoryItem.from_row(r) for r in rows]

    # ========== 固化（Consolidation）==========

    def consolidate(self, similarity_threshold=0.3, min_episodes=3):
        """情景记忆 → 语义记忆固化
        找到相似的情景记忆簇，提取共性，生成语义记忆。
        模拟人脑海马体→皮层固化过程。
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            episodes = conn.execute(
                "SELECT * FROM memories WHERE type = 'episodic' AND expired = 0 ORDER BY timestamp"
            ).fetchall()
            conn.close()

            if len(episodes) < min_episodes:
                return {"consolidated": 0, "reason": "not enough episodes"}

            # 聚类：用相似度把情景记忆分组
            episode_items = [MemoryItem.from_row(r) for r in episodes]
            clusters = []  # [(center_idx, [member_indices])]
            used = set()

            for i, item in enumerate(episode_items):
                if i in used:
                    continue
                cluster = [i]
                used.add(i)
                for j in range(i + 1, len(episode_items)):
                    if j in used:
                        continue
                    sim = SimpleEmbedding.cosine_sim(item.embedding, episode_items[j].embedding)
                    if sim > similarity_threshold:
                        cluster.append(j)
                        used.add(j)
                clusters.append(cluster)

            # 从每个簇提取语义记忆
            consolidated_count = 0
            for cluster in clusters:
                if len(cluster) < min_episodes:
                    continue
                members = [episode_items[i] for i in cluster]
                # 提取共性内容：取最重要成员的 content 作为摘要
                members.sort(key=lambda x: x.importance, reverse=True)
                summary_content = members[0].content
                avg_importance = sum(m.importance for m in members) / len(members)
                avg_confidence = sum(m.confidence for m in members) / len(members)
                source_ids = ",".join(m.id for m in members)
                tags_union = list(set(t for m in members for t in m.tags))

                self.store_semantic(
                    content=f"[固化] {summary_content} (来源: {len(members)} 条情景记忆)",
                    source="consolidation",
                    importance=min(avg_importance + 0.1, 1.0),  # 固化后重要性略升
                    tags=tags_union,
                    confidence=avg_confidence,
                    evidence="HYPOTHESIS",
                    consolidated_from=source_ids
                )
                consolidated_count += 1

            return {"consolidated": consolidated_count, "clusters": len(clusters)}

    # ========== 遗忘 ==========

    def forget(self, max_age_days=90, min_importance=0.3, max_count=10000):
        """基于重要性和时间的遗忘机制
        低重要性 + 老旧 + 低访问 → 过期
        """
        with self.lock:
            now = time.time()
            max_age = max_age_days * 86400
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # 统计总量
            total = conn.execute("SELECT COUNT(*) as c FROM memories WHERE expired = 0").fetchone()["c"]

            # 过期低重要性老旧记忆
            expired_count = conn.execute("""
                UPDATE memories SET expired = 1
                WHERE expired = 0
                  AND importance < ?
                  AND ( ? - timestamp ) > ?
                  AND access_count < 3
            """, (min_importance, now, max_age)).rowcount

            # 如果总量仍超限，按重要性淘汰最不重要的
            if total > max_count:
                conn.execute("""
                    UPDATE memories SET expired = 1
                    WHERE id IN (
                        SELECT id FROM memories WHERE expired = 0
                        ORDER BY importance ASC, access_count ASC LIMIT ?
                    )
                """, (total - max_count,))

            conn.commit()
            conn.close()
            return {"expired": expired_count, "total_before": total}

    # ========== 冲突解决 ==========

    def resolve_conflicts(self, similarity_threshold=0.3):
        """检测并标记知识冲突
        找到语义记忆中内容相似但证据等级冲突的条目。
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            semantics = conn.execute(
                "SELECT * FROM memories WHERE type = 'semantic' AND expired = 0"
            ).fetchall()
            conn.close()

            items = [MemoryItem.from_row(r) for r in semantics]
            conflicts = []

            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    sim = SimpleEmbedding.cosine_sim(items[i].embedding, items[j].embedding)
                    if sim > similarity_threshold:  # 高相似度
                        # 但证据等级冲突
                        ev_set = {items[i].evidence, items[j].evidence}
                        if ev_set & {"FACT", "VERIFIED"} and ev_set & {"FAILED", "UNCERTAIN"}:
                            conflicts.append({
                                "item_a": items[i].id,
                                "item_b": items[j].id,
                                "similarity": sim,
                                "evidence_a": items[i].evidence,
                                "evidence_b": items[j].evidence,
                                "content_a": items[i].content[:100],
                                "content_b": items[j].content[:100]
                            })

            return conflicts

    # ========== 统计 ==========

    def stats(self):
        """记忆系统统计"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            stats = {}
            for t in ["working", "episodic", "semantic", "procedural"]:
                row = conn.execute(
                    "SELECT COUNT(*) as c FROM memories WHERE type = ? AND expired = 0", (t,)
                ).fetchone()
                stats[t] = row["c"]
            row = conn.execute("SELECT COUNT(*) as c FROM memories WHERE expired = 1").fetchone()
            stats["expired"] = row["c"]
            conn.close()
            stats["working_in_memory"] = len(self._working)
            return stats

    # ========== 持久化工作记忆 ==========

    def flush_working_to_episodic(self):
        """将工作记忆刷入情景记忆（会话结束时调用）"""
        with self.lock:
            count = 0
            for item in self._working:
                if item.importance > 0.3:  # 只保存有意义的
                    self.store_episodic(
                        item.content, source="working_flush",
                        importance=item.importance, tags=item.tags
                    )
                    count += 1
            self._working.clear()
            return count