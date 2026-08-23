"""
AGI Memory System v0.1
=====================
Hierarchical memory: working -> episodic -> semantic -> procedural
Supports: store, retrieve, consolidation (episodic->semantic), compression, forgetting, conflict resolution

Dependencies: standard library + sqlite3 (built-in)
Design principles:
  - No external dependencies, runs anywhere
  - All knowledge retains source, time, confidence, evidence level
  - Active consolidation (episodic -> semantic)
  - Importance-based forgetting
  - Thread-safe

Usage:
  from memory_system import MemorySystem
  mem = MemorySystem("./memory_store")
  mem.store_episodic("user asked about AGI architecture", source="conversation", importance=0.8)
  results = mem.retrieve("AGI architecture", top_k=5)
  mem.consolidate()  # episodic -> semantic
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
    """A single memory item."""
    id: str = ""
    type: str = "episodic"          # working/episodic/semantic/procedural
    content: str = ""               # memory content
    source: str = ""                # origin
    timestamp: float = 0.0          # creation time
    last_access: float = 0.0        # last access time
    access_count: int = 0           # access count
    importance: float = 0.5          # importance 0-1
    confidence: float = 0.5         # confidence 0-1
    evidence: str = "UNKNOWN"       # evidence level
    tags: list = field(default_factory=list)
    embedding: list = field(default_factory=list)  # simple word-freq vector
    consolidated_from: str = ""     # consolidation source (episodic memory IDs)
    version: int = 1                # version number
    expired: bool = False           # is expired

    def to_dict(self):
        d = asdict(self)
        d["tags"] = json.dumps(self.tags, ensure_ascii=False)
        d["embedding"] = json.dumps(self.embedding)
        return d

    @classmethod
    def from_row(cls, row):
        """Construct from database row."""
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
    """Simple word-frequency vector -- no external dependency.
    Uses TF (term frequency) for vectorization, cosine similarity for retrieval.
    Good enough for testing. Can be replaced with real embedding later.
    """

    def __init__(self):
        self.vocab = {}  # word -> index
        self.lock = threading.Lock()

    def _tokenize(self, text):
        """Simple tokenization: English words, Chinese chars + bigrams.
        Chinese bigrams improve short-text similarity matching.
        """
        import re
        tokens = re.findall(r'[a-zA-Z]+', text.lower())
        cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens += cn_chars
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
        return [[k, v] for k, v in sorted(vec.items())]

    @staticmethod
    def cosine_sim(v1, v2):
        """Sparse vector cosine similarity."""
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
    """AGI hierarchical memory system."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self.db_path = os.path.join(store_path, "memory.db")
        self.embedder = SimpleEmbedding()
        self.lock = threading.RLock()

        # Working memory (in-memory, not persisted)
        self._working: list[MemoryItem] = []
        self._working_max = 20  # working memory capacity

        self._init_db()

    def _init_db(self):
        """Initialize database."""
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

    # ========== Store ==========

    def store_working(self, content, importance=0.5, tags=None):
        """Store in working memory (in-memory)."""
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
            if len(self._working) > self._working_max:
                self._working.sort(key=lambda x: x.importance * x.access_count)
                evicted = self._working.pop(0)
                if evicted.importance > 0.5:
                    self.store_episodic(evicted.content, source="working_memory_overflow",
                                       importance=evicted.importance, tags=evicted.tags)
            return item.id

    def store_episodic(self, content, source="", importance=0.5, tags=None,
                       confidence=0.5, evidence="UNKNOWN"):
        """Store in episodic memory (persisted)."""
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
        """Store in semantic memory (persisted, usually consolidation result)."""
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
        """Store in procedural memory (skills/methods)."""
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

    # ========== Retrieve ==========

    def retrieve(self, query, top_k=5, memory_types=None, min_importance=0.0):
        """Semantic retrieval: find top_k most relevant memories."""
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
                age = time.time() - item.timestamp
                time_decay = 1.0 / (1.0 + age / 86400)  # daily decay
                score = sim * item.importance * (0.5 + 0.5 * time_decay)
                results.append((score, item))

            results.sort(key=lambda x: x[0], reverse=True)
            results = results[:top_k]

            for score, item in results:
                conn.execute("""
                    UPDATE memories SET last_access = ?, access_count = access_count + 1
                    WHERE id = ?
                """, (time.time(), item.id))
            conn.commit()
            conn.close()

            return [(score, item) for score, item in results]

    def get_working(self):
        """Get working memory items."""
        with self.lock:
            return list(self._working)

    def get_all(self, memory_type=None, limit=100):
        """Get all memories of a given type."""
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

    # ========== Consolidation ==========

    def consolidate(self, similarity_threshold=0.3, min_episodes=3):
        """Episodic -> semantic consolidation.
        Find similar episodic memory clusters, extract commonalities, generate semantic memories.
        Simulates hippocampus -> cortex consolidation.
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

            episode_items = [MemoryItem.from_row(r) for r in episodes]
            clusters = []
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

            consolidated_count = 0
            for cluster in clusters:
                if len(cluster) < min_episodes:
                    continue
                members = [episode_items[i] for i in cluster]
                members.sort(key=lambda x: x.importance, reverse=True)
                summary_content = members[0].content
                avg_importance = sum(m.importance for m in members) / len(members)
                avg_confidence = sum(m.confidence for m in members) / len(members)
                source_ids = ",".join(m.id for m in members)
                tags_union = list(set(t for m in members for t in m.tags))

                self.store_semantic(
                    content=f"[consolidated] {summary_content} (from {len(members)} episodes)",
                    source="consolidation",
                    importance=min(avg_importance + 0.1, 1.0),
                    tags=tags_union,
                    confidence=avg_confidence,
                    evidence="HYPOTHESIS",
                    consolidated_from=source_ids
                )
                consolidated_count += 1

            return {"consolidated": consolidated_count, "clusters": len(clusters)}

    # ========== Forgetting ==========

    def forget(self, max_age_days=90, min_importance=0.3, max_count=10000):
        """Importance-and-time-based forgetting.
        Low importance + old + low access -> expired.
        """
        with self.lock:
            now = time.time()
            max_age = max_age_days * 86400
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            total = conn.execute("SELECT COUNT(*) as c FROM memories WHERE expired = 0").fetchone()["c"]

            expired_count = conn.execute("""
                UPDATE memories SET expired = 1
                WHERE expired = 0
                  AND importance < ?
                  AND ( ? - timestamp ) > ?
                  AND access_count < 3
            """, (min_importance, now, max_age)).rowcount

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

    # ========== Conflict Resolution ==========

    def resolve_conflicts(self, similarity_threshold=0.3):
        """Detect and flag knowledge conflicts.
        Find semantic memories with similar content but conflicting evidence levels.
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
                    if sim > similarity_threshold:
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

    # ========== Statistics ==========

    def stats(self):
        """Memory system statistics."""
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

    # ========== Flush Working Memory ==========

    def flush_working_to_episodic(self):
        """Flush working memory to episodic memory (call at session end)."""
        with self.lock:
            count = 0
            for item in self._working:
                if item.importance > 0.3:
                    self.store_episodic(
                        item.content, source="working_flush",
                        importance=item.importance, tags=item.tags
                    )
                    count += 1
            self._working.clear()
            return count
