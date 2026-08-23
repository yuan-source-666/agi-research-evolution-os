"""
AGI Continual Learning Module v0.1
==================================
Prevents catastrophic forgetting and enables skill accumulation.

Core mechanisms:
  1. Experience Replay Buffer -- stores task traces, replays to reinforce learning
  2. Skill Extraction -- analyzes successful traces, extracts reusable patterns
  3. Knowledge Consolidation -- merges episodic memories into semantic knowledge
  4. Anti-Forgetting Check -- periodic regression check via benchmark
  5. Learning Scheduler -- decides when to consolidate / replay / extract

Integration:
  - Reads TaskTraces from EvolutionLoop
  - Stores extracted skills to MemorySystem (procedural memory)
  - Uses BenchmarkSystem for regression detection
  - Reports to EvolutionLoop for evidence-based improvement

Dependencies: standard library
"""

import json
import time
import os
import hashlib
import threading
import traceback
from typing import Any, Optional, Callable, List, Dict
from dataclasses import dataclass, field, asdict
from enum import Enum


class SkillType(Enum):
    REASONING = "reasoning"
    TOOL_PATTERN = "tool_pattern"
    PROBLEM_SOLVING = "problem_solving"
    DOMAIN_KNOWLEDGE = "domain_knowledge"


class LearningPhase(Enum):
    IDLE = "idle"
    REPLAY = "replay"
    EXTRACT = "extract"
    CONSOLIDATE = "consolidate"
    REGRESSION_CHECK = "regression_check"


@dataclass
class ExperienceEntry:
    """A replay buffer entry from task execution."""
    id: str = ""
    trace_id: str = ""
    task_description: str = ""
    tools_used: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    success: bool = False
    metrics: dict = field(default_factory=dict)
    timestamp: float = 0.0
    replay_count: int = 0          # how many times replayed
    skill_extracted: bool = False   # whether a skill was extracted from this


@dataclass
class Skill:
    """An extracted reusable skill."""
    id: str = ""
    name: str = ""
    type: str = "problem_solving"
    description: str = ""
    pattern: str = ""               # abstract pattern description
    trigger_conditions: list = field(default_factory=list)  # when to apply
    action_template: str = ""       # how to execute
    success_rate: float = 0.0       # historical success rate
    usage_count: int = 0
    source_traces: list = field(default_factory=list)  # trace IDs that contributed
    created_at: float = 0.0
    last_used: float = 0.0
    confidence: float = 0.5
    evidence: str = "HYPOTHESIS"


@dataclass
class LearningReport:
    """Report from a learning cycle."""
    phase: str = ""
    skills_extracted: int = 0
    memories_consolidated: int = 0
    experiences_replayed: int = 0
    regression_detected: bool = False
    score_before: float = 0.0
    score_after: float = 0.0
    skill_library_size: int = 0
    timestamp: float = 0.0
    details: list = field(default_factory=list)


class ExperienceReplayBuffer:
    """Stores and replays task execution experiences."""

    def __init__(self, max_size: int = 1000):
        self.buffer: List[ExperienceEntry] = []
        self.max_size = max_size
        self.lock = threading.RLock()

    def add(self, trace_id: str, task_description: str, tools_used: list,
            steps: list, success: bool, metrics: dict):
        """Add a new experience to the buffer."""
        with self.lock:
            entry = ExperienceEntry(
                id=hashlib.md5(f"{trace_id}{time.time()}".encode()).hexdigest()[:12],
                trace_id=trace_id,
                task_description=task_description,
                tools_used=tools_used,
                steps=steps,
                success=success,
                metrics=metrics,
                timestamp=time.time(),
            )
            self.buffer.append(entry)
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)  # FIFO eviction
            return entry.id

    def sample(self, n: int = 10, success_only: bool = False) -> List[ExperienceEntry]:
        """Sample n experiences from buffer. Prioritize successful + recent."""
        with self.lock:
            pool = self.buffer
            if success_only:
                # 70% successful, 30% failed (to learn from mistakes too)
                successes = [e for e in pool if e.success]
                failures = [e for e in pool if not e.success]
                n_success = int(n * 0.7)
                n_failure = n - n_success
                import random
                sampled = random.sample(successes, min(n_success, len(successes)))
                sampled += random.sample(failures, min(n_failure, len(failures)))
                return sampled
            return pool[-n:]  # most recent

    def stats(self) -> dict:
        with self.lock:
            return {
                "total": len(self.buffer),
                "successful": sum(1 for e in self.buffer if e.success),
                "failed": sum(1 for e in self.buffer if not e.success),
                "avg_replay_count": (
                    sum(e.replay_count for e in self.buffer) / max(len(self.buffer), 1)
                ),
            }


class SkillExtractor:
    """Extracts reusable skills from successful task traces."""

    def __init__(self):
        self.lock = threading.RLock()

    def extract_from_traces(self, experiences: List[ExperienceEntry],
                            min_success_count: int = 2) -> List[Skill]:
        """Extract skills from a set of experiences.
        Groups similar successful traces and creates a Skill for each group.
        """
        with self.lock:
            # Only consider successful experiences
            successful = [e for e in experiences if e.success]
            if len(successful) < min_success_count:
                return []

            # Group by tool usage patterns
            tool_groups = {}
            for exp in successful:
                tool_key = tuple(sorted(set(exp.tools_used)))
                if tool_key not in tool_groups:
                    tool_groups[tool_key] = []
                tool_groups[tool_key].append(exp)

            skills = []
            for tool_combo, group in tool_groups.items():
                if len(group) < min_success_count:
                    continue

                # Create skill from group
                tool_names = list(tool_combo) if tool_combo else ["none"]
                skill_name = f"skill_{tool_names[0]}_{len(group)}"

                # Extract common steps pattern
                common_steps = self._find_common_steps(group)

                skill = Skill(
                    id=hashlib.md5(f"{skill_name}{time.time()}".encode()).hexdigest()[:12],
                    name=skill_name,
                    type=SkillType.TOOL_PATTERN.value if tool_combo else SkillType.REASONING.value,
                    description=f"Pattern using tools: {', '.join(tool_names)}. "
                               f"Success rate from {len(group)} traces.",
                    pattern=f"Tools: {', '.join(tool_names)} -> {common_steps}",
                    trigger_conditions=self._extract_triggers(group),
                    action_template=common_steps,
                    success_rate=len(group) / len(experiences),
                    source_traces=[e.trace_id for e in group],
                    created_at=time.time(),
                    confidence=min(0.5 + 0.1 * len(group), 0.95),
                    evidence="VERIFIED" if len(group) >= 3 else "HYPOTHESIS",
                )
                skills.append(skill)

            return skills

    def _find_common_steps(self, group: List[ExperienceEntry]) -> str:
        """Find common action patterns across experiences in a group."""
        if not group:
            return ""

        # Simple approach: list the tools used in order
        first = group[0]
        steps_desc = []
        for step in first.steps[:5]:  # limit to first 5 steps
            if isinstance(step, dict):
                action = step.get("action", step.get("finding", ""))
                steps_desc.append(str(action)[:80])
        return " -> ".join(steps_desc) if steps_desc else "execute and verify"

    def _extract_triggers(self, group: List[ExperienceEntry]) -> list:
        """Extract common trigger conditions from experiences."""
        # Simple keyword extraction from task descriptions
        word_freq = {}
        for exp in group:
            words = exp.task_description.lower().split()
            for w in words:
                if len(w) > 3:
                    word_freq[w] = word_freq.get(w, 0) + 1

        # Top 5 most common words as triggers
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, c in sorted_words[:5]]


class ContinualLearning:
    """Main continual learning coordinator."""

    def __init__(self, memory=None, evolution=None, benchmark_fn: Optional[Callable] = None,
                 log_dir: str = None):
        self.memory = memory
        self.evolution = evolution
        self.benchmark_fn = benchmark_fn  # callable that runs benchmark
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "CL_LOGS"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.replay_buffer = ExperienceReplayBuffer(max_size=500)
        self.skill_extractor = SkillExtractor()
        self.skill_library: Dict[str, Skill] = {}

        self.phase = LearningPhase.IDLE
        self.lock = threading.RLock()
        self.learning_history: List[LearningReport] = []

        # Load existing skills from memory if available
        self._load_skills_from_memory()

    def _load_skills_from_memory(self):
        """Load existing procedural memories as skills."""
        if not self.memory:
            return
        try:
            procedural = self.memory.get_all("procedural", limit=100)
            for item in procedural:
                skill = Skill(
                    id=item.id,
                    name=f"loaded_skill_{item.id[:8]}",
                    type=SkillType.DOMAIN_KNOWLEDGE.value,
                    description=item.content[:200],
                    pattern=item.content,
                    created_at=item.timestamp,
                    confidence=item.confidence,
                    evidence=item.evidence,
                )
                self.skill_library[skill.id] = skill
        except Exception as e:
            print(f"[ContinualLearning] Load skills failed: {e}", flush=True)

    def record_experience(self, trace_id: str, task_description: str,
                          tools_used: list, steps: list, success: bool, metrics: dict):
        """Record a new experience from task execution."""
        self.replay_buffer.add(
            trace_id=trace_id, task_description=task_description,
            tools_used=tools_used, steps=steps, success=success, metrics=metrics
        )

    def run_learning_cycle(self) -> LearningReport:
        """Run a full continual learning cycle:
        1. Replay experiences
        2. Extract skills
        3. Consolidate memories
        4. Regression check
        """
        report = LearningReport(timestamp=time.time())
        details = []

        # Phase 1: Experience Replay
        self.phase = LearningPhase.REPLAY
        sampled = self.replay_buffer.sample(n=10, success_only=True)
        replayed = 0
        for exp in sampled:
            exp.replay_count += 1
            replayed += 1
            # Store replayed experience to working memory
            if self.memory:
                self.memory.store_working(
                    f"Replay: {exp.task_description[:100]} -> {'ok' if exp.success else 'fail'}",
                    importance=0.3
                )
        report.experiences_replayed = replayed
        details.append(f"Replayed {replayed} experiences")

        # Phase 2: Skill Extraction
        self.phase = LearningPhase.EXTRACT
        new_skills = self.skill_extractor.extract_from_traces(
            self.replay_buffer.buffer, min_success_count=2
        )
        for skill in new_skills:
            if skill.id not in self.skill_library:
                self.skill_library[skill.id] = skill
                # Store to procedural memory
                if self.memory:
                    self.memory.store_procedural(
                        content=f"Skill: {skill.name} -- {skill.description} "
                               f"(pattern: {skill.pattern})",
                        source="continual_learning",
                        importance=skill.confidence,
                        tags=["skill", "extracted", skill.type],
                        confidence=skill.confidence,
                        evidence=skill.evidence,
                    )
        report.skills_extracted = len(new_skills)
        details.append(f"Extracted {len(new_skills)} new skills")
        report.skill_library_size = len(self.skill_library)

        # Phase 3: Memory Consolidation
        self.phase = LearningPhase.CONSOLIDATE
        consolidated = 0
        if self.memory:
            try:
                result = self.memory.consolidate()
                consolidated = result.get("consolidated", 0)
            except Exception as e:
                details.append(f"Consolidation error: {str(e)[:100]}")
        report.memories_consolidated = consolidated
        details.append(f"Consolidated {consolidated} memory clusters")

        # Phase 4: Regression Check (optional, if benchmark_fn provided)
        self.phase = LearningPhase.REGRESSION_CHECK
        if self.benchmark_fn:
            try:
                bench_result = self.benchmark_fn()
                report.score_after = bench_result.get("overall_score", 0)
                report.regression_detected = bench_result.get("overall_score", 1) < 0.5
                details.append(f"Benchmark score: {report.score_after}")
            except Exception as e:
                details.append(f"Benchmark error: {str(e)[:100]}")

        self.phase = LearningPhase.IDLE
        report.phase = "complete"
        report.details = details

        # Store to history
        with self.lock:
            self.learning_history.append(report)

        # Log
        self._log("learning_cycle", json.dumps(asdict(report), default=str))

        print(f"[ContinualLearning] Cycle complete: "
              f"{report.skills_extracted} skills, "
              f"{report.memories_consolidated} consolidated, "
              f"{report.experiences_replayed} replayed", flush=True)

        return report

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        with self.lock:
            return self.skill_library.get(skill_id)

    def find_relevant_skills(self, task_description: str, top_k: int = 3) -> List[Skill]:
        """Find skills relevant to a task (simple keyword matching)."""
        with self.lock:
            task_words = set(task_description.lower().split())
            scored = []
            for skill in self.skill_library.values():
                trigger_words = set(skill.trigger_conditions)
                overlap = len(task_words & trigger_words)
                scored.append((overlap + skill.usage_count * 0.1, skill))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [s for _, s in scored[:top_k]]

    def stats(self) -> dict:
        with self.lock:
            return {
                "phase": self.phase.value,
                "replay_buffer": self.replay_buffer.stats(),
                "skill_library_size": len(self.skill_library),
                "learning_cycles": len(self.learning_history),
                "skills": [
                    {"name": s.name, "type": s.type, "success_rate": s.success_rate,
                     "confidence": s.confidence, "usage_count": s.usage_count}
                    for s in list(self.skill_library.values())[:20]
                ],
            }

    def _log(self, event: str, message: str):
        log_path = os.path.join(self.log_dir, "continual_learning_log.jsonl")
        entry = {"timestamp": time.time(), "event": event, "message": message}
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
