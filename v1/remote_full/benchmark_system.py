"""
AGI Benchmark System v0.1
=========================
Standardized evaluation suite for AGI capabilities.
Provides quantitative metrics for reasoning, tool use, memory, code execution, and planning.

Design:
  - TaskSuite: collection of standardized tasks with expected outcomes
  - TaskRunner: executes tasks against the AGI system, measures metrics
  - ScoreReport: aggregated scores with per-task breakdown
  - RegressionTracker: compare scores across versions to detect degradation
  - Integration: feeds results to EvolutionLoop for evidence-based improvement

Dependencies: standard library + requests (for calling AGI core API)
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


class TaskCategory(Enum):
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    MEMORY = "memory"
    CODE_EXEC = "code_exec"
    PLANNING = "planning"
    MULTI_STEP = "multi_step"


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    id: str
    category: str
    difficulty: int
    prompt: str
    expected_keywords: list = field(default_factory=list)  # keywords expected in response
    expected_tools: list = field(default_factory=list)     # tools expected to be used
    timeout: int = 120
    max_rounds: int = 5
    scoring_fn: str = "keyword_match"  # scoring function name
    weight: float = 1.0


@dataclass
class TaskResult:
    """Result of running a single benchmark task."""
    task_id: str
    category: str
    difficulty: int
    response: str = ""
    tools_used: list = field(default_factory=list)
    success: bool = False
    score: float = 0.0
    latency: float = 0.0
    rounds: int = 0
    error: str = ""


@dataclass
class ScoreReport:
    """Aggregated benchmark scores."""
    total_tasks: int = 0
    completed: int = 0
    succeeded: int = 0
    overall_score: float = 0.0
    category_scores: dict = field(default_factory=dict)
    difficulty_scores: dict = field(default_factory=dict)
    avg_latency: float = 0.0
    task_results: list = field(default_factory=list)
    timestamp: float = 0.0
    version_id: str = ""


class TaskSuite:
    """Collection of benchmark tasks."""

    def __init__(self):
        self.tasks: Dict[str, BenchmarkTask] = {}
        self._init_default_tasks()

    def _init_default_tasks(self):
        """Initialize default benchmark task suite."""
        # --- Reasoning tasks ---
        self.add(BenchmarkTask(
            id="reason_001", category=TaskCategory.REASONING.value,
            difficulty=Difficulty.EASY.value,
            prompt="What is 15 + 27? Answer with just the number.",
            expected_keywords=["42"],
            scoring_fn="exact_match", weight=1.0, timeout=60
        ))
        self.add(BenchmarkTask(
            id="reason_002", category=TaskCategory.REASONING.value,
            difficulty=Difficulty.MEDIUM.value,
            prompt="If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly? Explain your reasoning.",
            expected_keywords=["no", "cannot", "not necessarily", "invalid"],
            scoring_fn="keyword_match", weight=1.5, timeout=90
        ))
        self.add(BenchmarkTask(
            id="reason_003", category=TaskCategory.REASONING.value,
            difficulty=Difficulty.HARD.value,
            prompt="Solve this logic puzzle: Three people (A, B, C) each have a different job (doctor, teacher, engineer). A is not the doctor. B is not the teacher. The doctor is older than B. Who is the engineer? Explain step by step.",
            expected_keywords=["a", "teacher", "c", "doctor", "engineer"],
            scoring_fn="keyword_match", weight=2.0, timeout=120
        ))

        # --- Tool use tasks ---
        self.add(BenchmarkTask(
            id="tool_001", category=TaskCategory.TOOL_USE.value,
            difficulty=Difficulty.EASY.value,
            prompt="Use code_exec to calculate the factorial of 10. Report the result.",
            expected_keywords=["3628800"],
            expected_tools=["code_exec"],
            scoring_fn="keyword_match", weight=1.0, timeout=120
        ))
        self.add(BenchmarkTask(
            id="tool_002", category=TaskCategory.TOOL_USE.value,
            difficulty=Difficulty.MEDIUM.value,
            prompt="Use code_exec to generate the first 20 Fibonacci numbers. List them all.",
            expected_keywords=["0", "1", "6765"],
            expected_tools=["code_exec"],
            scoring_fn="keyword_match", weight=1.5, timeout=120
        ))

        # --- Memory tasks ---
        self.add(BenchmarkTask(
            id="mem_001", category=TaskCategory.MEMORY.value,
            difficulty=Difficulty.EASY.value,
            prompt="Store this fact in your memory: 'The capital of Mars colony will be Olympus City.' Then confirm you stored it.",
            expected_keywords=["stored", "memory", "confirmed"],
            expected_tools=["memory_store"],
            scoring_fn="keyword_match", weight=1.0, timeout=90
        ))

        # --- Code execution tasks ---
        self.add(BenchmarkTask(
            id="code_001", category=TaskCategory.CODE_EXEC.value,
            difficulty=Difficulty.MEDIUM.value,
            prompt="Write and execute Python code to check if 997 is a prime number. Report the result.",
            expected_keywords=["prime", "997", "yes", "true"],
            expected_tools=["code_exec"],
            scoring_fn="keyword_match", weight=1.5, timeout=120
        ))
        self.add(BenchmarkTask(
            id="code_002", category=TaskCategory.CODE_EXEC.value,
            difficulty=Difficulty.HARD.value,
            prompt="Write and execute Python code to sort the list [5, 2, 8, 1, 9, 3] using quicksort. Show the sorted result.",
            expected_keywords=["1", "2", "3", "5", "8", "9", "sorted"],
            expected_tools=["code_exec"],
            scoring_fn="keyword_match", weight=2.0, timeout=120
        ))

        # --- Planning tasks ---
        self.add(BenchmarkTask(
            id="plan_001", category=TaskCategory.PLANNING.value,
            difficulty=Difficulty.MEDIUM.value,
            prompt="Plan a 3-step research approach to investigate renewable energy storage solutions. List the steps clearly.",
            expected_keywords=["step", "1", "2", "3", "research", "energy", "storage"],
            scoring_fn="keyword_match", weight=1.5, timeout=90
        ))

    def add(self, task: BenchmarkTask):
        self.tasks[task.id] = task

    def get_all(self) -> List[BenchmarkTask]:
        return list(self.tasks.values())

    def get_by_category(self, category: str) -> List[BenchmarkTask]:
        return [t for t in self.tasks.values() if t.category == category]

    def count(self) -> int:
        return len(self.tasks)


class ScoringFunctions:
    """Built-in scoring functions."""

    @staticmethod
    def keyword_match(response: str, expected_keywords: list) -> float:
        """Check if expected keywords appear in response. Returns 0-1 score."""
        if not expected_keywords:
            return 1.0
        response_lower = response.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        return matched / len(expected_keywords)

    @staticmethod
    def exact_match(response: str, expected_keywords: list) -> float:
        """Check if response contains exact match of any expected keyword."""
        if not expected_keywords:
            return 1.0
        response_stripped = response.strip().lower()
        for kw in expected_keywords:
            if kw.lower() == response_stripped:
                return 1.0
            if kw.lower() in response_stripped:
                return 0.8
        return 0.0

    @staticmethod
    def tool_usage_score(tools_used: list, expected_tools: list) -> float:
        """Check if expected tools were used."""
        if not expected_tools:
            return 1.0
        matched = sum(1 for t in expected_tools if t in tools_used)
        return matched / len(expected_tools)

    @staticmethod
    def get_fn(name: str) -> Callable:
        fns = {
            "keyword_match": ScoringFunctions.keyword_match,
            "exact_match": ScoringFunctions.exact_match,
        }
        return fns.get(name, ScoringFunctions.keyword_match)


class TaskRunner:
    """Executes benchmark tasks against the AGI system."""

    def __init__(self, agi_core_url: str = "http://127.0.0.1:9099"):
        self.agi_core_url = agi_core_url.rstrip("/")
        self.suite = TaskSuite()
        self.lock = threading.RLock()
        self.results_history: List[ScoreReport] = []

    def run_task(self, task: BenchmarkTask) -> TaskResult:
        """Run a single benchmark task against AGI core."""
        import requests as req

        result = TaskResult(
            task_id=task.id,
            category=task.category,
            difficulty=task.difficulty,
        )

        t_start = time.time()
        try:
            payload = {
                "message": task.prompt,
                "max_rounds": task.max_rounds,
            }
            resp = req.post(
                f"{self.agi_core_url}/chat",
                json=payload,
                timeout=task.timeout
            )
            latency = time.time() - t_start

            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                result.latency = latency
                return result

            data = resp.json()
            result.response = data.get("response", "")
            result.tools_used = [tc["name"] for tc in data.get("tool_calls_log", [])]
            result.rounds = data.get("rounds", 0)
            result.latency = round(latency, 2)

            # Score the response
            score_fn = ScoringFunctions.get_fn(task.scoring_fn)
            content_score = score_fn(result.response, task.expected_keywords)
            tool_score = ScoringFunctions.tool_usage_score(result.tools_used, task.expected_tools)

            # Combined score: 70% content + 30% tool usage
            result.score = round(content_score * 0.7 + tool_score * 0.3, 4)
            result.success = result.score >= 0.6

        except Exception as e:
            result.latency = round(time.time() - t_start, 2)
            result.error = str(e)[:200]

        return result

    def run_suite(self, categories: Optional[List[str]] = None,
                  task_ids: Optional[List[str]] = None,
                  version_id: str = "") -> ScoreReport:
        """Run full benchmark suite (or subset). Returns aggregated report."""
        tasks = self.suite.get_all()
        if categories:
            tasks = [t for t in tasks if t.category in categories]
        if task_ids:
            tasks = [t for t in tasks if t.id in task_ids]

        report = ScoreReport(
            total_tasks=len(tasks),
            timestamp=time.time(),
            version_id=version_id,
        )

        results = []
        for i, task in enumerate(tasks):
            print(f"[Benchmark] Running {i+1}/{len(tasks)}: {task.id} ({task.category})", flush=True)
            result = self.run_task(task)
            results.append(result)
            print(f"  -> score={result.score}, success={result.success}, "
                  f"latency={result.latency}s, tools={result.tools_used}", flush=True)

        # Aggregate
        report.task_results = [asdict(r) for r in results]
        report.completed = sum(1 for r in results if not r.error)
        report.succeeded = sum(1 for r in results if r.success)
        report.avg_latency = round(
            sum(r.latency for r in results) / max(len(results), 1), 2
        )

        # Category scores
        cat_scores = {}
        for cat in TaskCategory:
            cat_results = [r for r in results if r.category == cat.value]
            if cat_results:
                weighted = sum(r.score * 1.0 for r in cat_results)
                cat_scores[cat.value] = round(weighted / len(cat_results), 4)
        report.category_scores = cat_scores

        # Difficulty scores
        diff_scores = {}
        for diff in Difficulty:
            diff_results = [r for r in results if r.difficulty == diff.value]
            if diff_results:
                diff_scores[f"level_{diff.value}"] = round(
                    sum(r.score for r in diff_results) / len(diff_results), 4
                )
        report.difficulty_scores = diff_scores

        # Overall score
        if results:
            report.overall_score = round(
                sum(r.score for r in results) / len(results), 4
            )

        # Store history
        with self.lock:
            self.results_history.append(report)

        return report

    def get_history(self, n: int = 10) -> list:
        """Get recent benchmark reports."""
        with self.lock:
            return [asdict(r) for r in self.results_history[-n:]]

    def compare_versions(self, report_a: ScoreReport, report_b: ScoreReport) -> dict:
        """Compare two benchmark reports to detect regression."""
        delta = report_b.overall_score - report_a.overall_score
        cat_delta = {}
        for cat in set(list(report_a.category_scores.keys()) + list(report_b.category_scores.keys())):
            a = report_a.category_scores.get(cat, 0)
            b = report_b.category_scores.get(cat, 0)
            cat_delta[cat] = round(b - a, 4)

        regression = delta < -0.05 or any(v < -0.1 for v in cat_delta.values())

        return {
            "overall_delta": round(delta, 4),
            "category_deltas": cat_delta,
            "regression_detected": regression,
            "latency_delta": round(report_b.avg_latency - report_a.avg_latency, 2),
        }


def run_benchmark(agi_core_url: str = "http://127.0.0.1:9099",
                  version_id: str = "",
                  categories: Optional[List[str]] = None) -> dict:
    """Convenience function to run benchmark and return dict."""
    runner = TaskRunner(agi_core_url)
    report = runner.run_suite(categories=categories, version_id=version_id)
    return asdict(report)
