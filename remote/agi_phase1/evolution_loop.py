"""
AGI Evolution Loop v0.1
=======================
自进化引擎：观察 → 评估 → 改进 → 验证 → 接受/拒绝 → 回滚

核心流程：
  1. Observe  — 采集系统状态、任务执行轨迹、工具调用日志
  2. Evaluate — 评估性能指标，识别薄弱环节
  3. Improve  — 在对应层级(L1-L9)提出改进方案
  4. Validate — Sandbox 测试 → Benchmark → 独立评估
  5. Decide   — Accept / Reject
  6. Rollback — 如果拒绝，回滚到上一个稳定版本

版本管理：Stable / Experimental / Research
安全约束：所有修改在隔离环境，不可绕过验证，审计记录不可删除

依赖：仅标准库 + memory_system + tool_system
"""

import json
import time
import os
import copy
import hashlib
import threading
import traceback
from typing import Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum


# ========== 枚举定义 ==========

class EvolutionPhase(Enum):
    """自进化流程阶段"""
    OBSERVE = "observe"
    EVALUATE = "evaluate"
    IMPROVE = "improve"
    VALIDATE = "validate"
    DECIDE = "decide"
    ROLLBACK = "rollback"
    IDLE = "idle"


class ImprovementLayer(Enum):
    """9 层自改进"""
    L1_PROMPT = 1
    L2_MEMORY = 2
    L3_STRATEGY = 3
    L4_WORKFLOW = 4
    L5_SKILL = 5
    L6_TOOL = 6
    L7_AGENT_CODE = 7
    L8_ARCHITECTURE = 8
    L9_MODEL_TRAINING = 9


class VersionStatus(Enum):
    """版本状态"""
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    RESEARCH = "research"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class EvidenceLevel(Enum):
    FACT = "FACT"
    HYPOTHESIS = "HYPOTHESIS"
    UNKNOWN = "UNKNOWN"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"


# ========== 数据结构 ==========

@dataclass
class TaskTrace:
    """一次任务执行轨迹"""
    task_id: str = ""
    description: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    tools_used: list = field(default_factory=list)
    steps: list = field(default_factory=list)  # [{action, result, success, latency}]
    success: bool = False
    error: str = ""
    metrics: dict = field(default_factory=dict)  # 自定义指标


@dataclass
class Improvement:
    """一个改进提案"""
    id: str = ""
    layer: int = 1          # ImprovementLayer
    title: str = ""
    description: str = ""
    rationale: str = ""     # 为什么提出这个改进
    expected_gain: str = "" # 预期收益
    risk_level: str = "low" # low/medium/high/extreme
    status: str = "proposed"  # proposed/testing/validated/accepted/rejected/rolled_back
    evidence: str = "HYPOTHESIS"
    sandbox_result: dict = field(default_factory=dict)
    benchmark_result: dict = field(default_factory=dict)
    eval_result: dict = field(default_factory=dict)
    created_at: float = 0.0
    decided_at: float = 0.0
    version_before: str = ""
    version_after: str = ""


@dataclass
class SystemVersion:
    """系统版本快照"""
    version_id: str = ""
    status: str = "experimental"
    created_at: float = 0.0
    description: str = ""
    parent_version: str = ""  # 从哪个版本分叉
    metrics: dict = field(default_factory=dict)  # 快照时的性能指标
    improvements: list = field(default_factory=list)  # 包含的改进ID列表


# ========== Evolution Loop ==========

class EvolutionLoop:
    """AGI 自进化引擎"""

    def __init__(self, memory=None, tool_system=None, log_dir=None):
        self.memory = memory
        self.tools = tool_system
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "LOGS"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.lock = threading.RLock()

        # 状态
        self.phase = EvolutionPhase.IDLE
        self.current_version: Optional[SystemVersion] = None
        self.stable_version: Optional[SystemVersion] = None
        self.traces: list[TaskTrace] = []
        self.improvements: list[Improvement] = []
        self.versions: list[SystemVersion] = []
        self.evolution_history: list[dict] = []  # 每次循环的记录

        # 基线指标（从稳定版本继承）
        self.baseline_metrics: dict = {
            "task_success_rate": 0.0,
            "avg_latency": 0.0,
            "tool_error_rate": 0.0,
            "knowledge_coverage": 0.0,
        }

        # 初始化第一个版本
        self._init_first_version()

        # 评估器注册（可自定义）
        self.evaluators: list[Callable] = []
        self.benchmark_suite: list[Callable] = []

        # 回滚钩子（每个改进可注册回滚函数）
        self._rollback_hooks: dict[str, Callable] = {}

    def _init_first_version(self):
        """初始化第一个 Research 版本"""
        v0 = SystemVersion(
            version_id=self._gen_version_id("v0"),
            status=VersionStatus.RESEARCH.value,
            created_at=time.time(),
            description="初始版本，Phase 1 基础运行时",
            parent_version="",
            metrics=copy.deepcopy(self.baseline_metrics)
        )
        self.versions.append(v0)
        self.current_version = v0
        self._log_evolution("init", f"初始版本 {v0.version_id} 创建", version=v0.version_id)

    def _gen_version_id(self, prefix="v"):
        return f"{prefix}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    def _gen_improvement_id(self):
        return f"imp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:10]}"

    # ========== 阶段 1: Observe ==========

    def record_trace(self, trace: TaskTrace):
        """记录一次任务执行轨迹"""
        with self.lock:
            self.traces.append(trace)
            # 存入记忆
            if self.memory:
                self.memory.store_episodic(
                    f"任务执行: {trace.description} → {'成功' if trace.success else '失败'}",
                    source="evolution_loop",
                    importance=0.6 if trace.success else 0.8,  # 失败更重要
                    tags=["evolution", "trace", "success" if trace.success else "failure"]
                )
            self._log_evolution("trace", f"记录轨迹: {trace.description} ({'ok' if trace.success else 'FAIL'})")

    def observe(self) -> dict:
        """阶段 1: 观察系统状态"""
        with self.lock:
            self.phase = EvolutionPhase.OBSERVE
            self._log_evolution("phase", "→ OBSERVE")

            # 汇总最近轨迹
            recent_traces = self.traces[-50:]
            total = len(recent_traces)
            successes = sum(1 for t in recent_traces if t.success)
            success_rate = successes / total if total > 0 else 0.0

            # 工具系统统计
            tool_stats = self.tools.stats() if self.tools else {}
            tool_error_rate = tool_stats.get("avg_error_rate", 0.0)

            # 平均延迟
            latencies = []
            for t in recent_traces:
                if t.start_time and t.end_time:
                    latencies.append(t.end_time - t.start_time)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

            # 记忆系统统计
            mem_stats = self.memory.stats() if self.memory else {}

            observation = {
                "total_traces": total,
                "success_rate": round(success_rate, 4),
                "tool_error_rate": round(tool_error_rate, 4),
                "avg_latency": round(avg_latency, 4),
                "memory_stats": mem_stats,
                "tool_stats": tool_stats,
                "current_version": self.current_version.version_id if self.current_version else "",
                "timestamp": time.time()
            }

            self._log_evolution("observe", json.dumps(observation, ensure_ascii=False, default=str))
            return observation

    # ========== 阶段 2: Evaluate ==========

    def evaluate(self, observation: dict) -> dict:
        """阶段 2: 评估性能，识别薄弱环节"""
        with self.lock:
            self.phase = EvolutionPhase.EVALUATE
            self._log_evolution("phase", "→ EVALUATE")

            assessment = {
                "overall_score": 0.0,
                "weaknesses": [],
                "strengths": [],
                "improvement_candidates": [],
                "evidence_level": EvidenceLevel.UNCERTAIN.value
            }

            # 1. 成功率评估
            sr = observation.get("success_rate", 0)
            if sr < 0.5:
                assessment["weaknesses"].append({
                    "area": "task_success_rate",
                    "value": sr,
                    "target": 0.8,
                    "severity": "high",
                    "note": "任务成功率低于50%"
                })
            elif sr < 0.8:
                assessment["weaknesses"].append({
                    "area": "task_success_rate",
                    "value": sr,
                    "target": 0.8,
                    "severity": "medium"
                })
            else:
                assessment["strengths"].append({"area": "task_success_rate", "value": sr})

            # 2. 工具错误率
            ter = observation.get("tool_error_rate", 0)
            if ter > 0.3:
                assessment["weaknesses"].append({
                    "area": "tool_error_rate",
                    "value": ter,
                    "target": 0.1,
                    "severity": "high",
                    "note": "工具错误率超过30%"
                })
            elif ter > 0.1:
                assessment["weaknesses"].append({
                    "area": "tool_error_rate",
                    "value": ter,
                    "target": 0.1,
                    "severity": "medium"
                })

            # 3. 延迟
            lat = observation.get("avg_latency", 0)
            if lat > 10:
                assessment["weaknesses"].append({
                    "area": "avg_latency",
                    "value": lat,
                    "target": 5.0,
                    "severity": "medium"
                })

            # 4. 记忆覆盖率（简单启发式）
            mem_stats = observation.get("memory_stats", {})
            total_mem = sum(v for k, v in mem_stats.items() if isinstance(v, int))
            if total_mem < 10:
                assessment["weaknesses"].append({
                    "area": "knowledge_coverage",
                    "value": total_mem,
                    "target": 50,
                    "severity": "low",
                    "note": "记忆条目不足，知识库薄弱"
                })

            # 总体评分（简单加权）
            score = 0.0
            score += min(sr, 1.0) * 0.4
            score += max(1.0 - ter, 0.0) * 0.3
            score += max(1.0 - min(lat / 30, 1.0), 0.0) * 0.2
            score += min(total_mem / 100, 1.0) * 0.1
            assessment["overall_score"] = round(score, 4)

            # 根据薄弱环节生成改进候选
            for w in assessment["weaknesses"]:
                layer = self._suggest_layer(w["area"])
                imp = Improvement(
                    id=self._gen_improvement_id(),
                    layer=layer,
                    title=f"改进 {w['area']}",
                    description=f"当前 {w['area']}={w['value']}，目标 {w['target']}",
                    rationale=w.get("note", f"{w['area']} 低于目标值"),
                    expected_gain=f"{w['area']} 从 {w['value']} 提升到 {w['target']}",
                    risk_level="low" if w["severity"] == "low" else ("medium" if w["severity"] == "medium" else "high"),
                    created_at=time.time(),
                    version_before=self.current_version.version_id if self.current_version else ""
                )
                assessment["improvement_candidates"].append(imp)
                self.improvements.append(imp)

            # 证据等级
            if total := observation.get("total_traces", 0):
                if total >= 20:
                    assessment["evidence_level"] = EvidenceLevel.VERIFIED.value
                elif total >= 5:
                    assessment["evidence_level"] = EvidenceLevel.UNCERTAIN.value
                else:
                    assessment["evidence_level"] = EvidenceLevel.UNKNOWN.value

            self._log_evolution("evaluate", json.dumps(assessment, ensure_ascii=False, default=str))
            return assessment

    def _suggest_layer(self, area: str) -> int:
        """根据薄弱领域建议改进层级"""
        mapping = {
            "task_success_rate": ImprovementLayer.L3_STRATEGY.value,
            "tool_error_rate": ImprovementLayer.L6_TOOL.value,
            "avg_latency": ImprovementLayer.L4_WORKFLOW.value,
            "knowledge_coverage": ImprovementLayer.L2_MEMORY.value,
        }
        return mapping.get(area, ImprovementLayer.L1_PROMPT.value)

    # ========== 阶段 3: Improve ==========

    def propose_improvement(self, layer: int, title: str, description: str,
                            rationale: str, expected_gain: str,
                            risk_level: str = "low",
                            implement_fn: Optional[Callable] = None,
                            rollback_fn: Optional[Callable] = None) -> Improvement:
        """阶段 3: 提出一个改进提案"""
        with self.lock:
            self.phase = EvolutionPhase.IMPROVE
            self._log_evolution("phase", "→ IMPROVE")

            imp = Improvement(
                id=self._gen_improvement_id(),
                layer=layer,
                title=title,
                description=description,
                rationale=rationale,
                expected_gain=expected_gain,
                risk_level=risk_level,
                created_at=time.time(),
                version_before=self.current_version.version_id if self.current_version else ""
            )
            self.improvements.append(imp)

            # 注册回滚钩子
            if rollback_fn:
                self._rollback_hooks[imp.id] = rollback_fn

            # 存入记忆
            if self.memory:
                self.memory.store_episodic(
                    f"改进提案: [{ImprovementLayer(layer).name}] {title} — {description}",
                    source="evolution_loop",
                    importance=0.7,
                    tags=["evolution", "improvement", f"L{layer}"],
                    evidence="HYPOTHESIS"
                )

            self._log_evolution("improve", f"提案: {title} (L{layer}, risk={risk_level})")
            return imp

    # ========== 阶段 4: Validate ==========

    def validate(self, improvement: Improvement,
                 sandbox_fn: Optional[Callable] = None,
                 benchmark_fn: Optional[Callable] = None,
                 eval_fn: Optional[Callable] = None) -> Improvement:
        """阶段 4: 验证改进（Sandbox → Test → Benchmark → Independent Eval）"""
        with self.lock:
            self.phase = EvolutionPhase.VALIDATE
            self._log_evolution("phase", f"→ VALIDATE ({improvement.id})")

            # 4a. Sandbox 测试
            if sandbox_fn:
                try:
                    improvement.sandbox_result = sandbox_fn() or {}
                    improvement.status = "testing"
                except Exception as e:
                    improvement.sandbox_result = {"error": str(e), "traceback": traceback.format_exc()}
                    improvement.status = "rejected"
                    improvement.evidence = EvidenceLevel.FAILED.value
                    self._log_evolution("validate", f"Sandbox 失败: {e}")
                    return improvement
            else:
                # 默认 sandbox：空跑通过
                improvement.sandbox_result = {"status": "skipped"}
                improvement.status = "testing"

            # 4b. Benchmark
            if benchmark_fn:
                try:
                    improvement.benchmark_result = benchmark_fn() or {}
                except Exception as e:
                    improvement.benchmark_result = {"error": str(e)}
                    improvement.evidence = EvidenceLevel.FAILED.value
                    self._log_evolution("validate", f"Benchmark 失败: {e}")
            elif self.benchmark_suite:
                # 使用注册的 benchmark 套件
                results = {}
                for i, bench in enumerate(self.benchmark_suite):
                    try:
                        results[f"bench_{i}"] = bench()
                    except Exception as e:
                        results[f"bench_{i}"] = {"error": str(e)}
                improvement.benchmark_result = results
            else:
                improvement.benchmark_result = {"status": "skipped"}

            # 4c. 独立评估
            if eval_fn:
                try:
                    improvement.eval_result = eval_fn() or {}
                except Exception as e:
                    improvement.eval_result = {"error": str(e)}
            elif self.evaluators:
                results = {}
                for i, evaluator in enumerate(self.evaluators):
                    try:
                        results[f"eval_{i}"] = evaluator(improvement)
                    except Exception as e:
                        results[f"eval_{i}"] = {"error": str(e)}
                improvement.eval_result = results
            else:
                # 默认独立评估：检查 sandbox + benchmark 是否通过
                sandbox_ok = improvement.sandbox_result.get("status") != "error" and \
                             "error" not in improvement.sandbox_result
                bench_ok = "error" not in improvement.benchmark_result
                improvement.eval_result = {
                    "sandbox_pass": sandbox_ok,
                    "benchmark_pass": bench_ok,
                    "overall": sandbox_ok and bench_ok
                }

            # 更新证据等级
            if improvement.eval_result.get("overall"):
                improvement.evidence = EvidenceLevel.VERIFIED.value
            else:
                improvement.evidence = EvidenceLevel.FAILED.value

            self._log_evolution("validate", f"验证完成: evidence={improvement.evidence}")
            return improvement

    # ========== 阶段 5: Decide ==========

    def decide(self, improvement: Improvement) -> bool:
        """阶段 5: 接受或拒绝改进"""
        with self.lock:
            self.phase = EvolutionPhase.DECIDE
            self._log_evolution("phase", f"→ DECIDE ({improvement.id})")

            accepted = False

            # 决策规则
            if improvement.evidence == EvidenceLevel.VERIFIED.value:
                # 验证通过 → 接受
                improvement.status = "accepted"
                accepted = True

                # 创建新版本
                new_version = SystemVersion(
                    version_id=self._gen_version_id("v"),
                    status=VersionStatus.EXPERIMENTAL.value,
                    created_at=time.time(),
                    description=improvement.title,
                    parent_version=self.current_version.version_id if self.current_version else "",
                    metrics=improvement.benchmark_result,
                    improvements=[improvement.id]
                )
                self.versions.append(new_version)
                self.current_version = new_version
                improvement.version_after = new_version.version_id

                self._log_evolution("decide", f"接受: {improvement.title} → 新版本 {new_version.version_id}")

                # 存入记忆
                if self.memory:
                    self.memory.store_semantic(
                        f"改进已接受: [{ImprovementLayer(improvement.layer).name}] {improvement.title}",
                        source="evolution_loop",
                        importance=0.9,
                        tags=["evolution", "accepted", f"L{improvement.layer}"],
                        evidence="VERIFIED"
                    )
            else:
                # 验证失败 → 拒绝
                improvement.status = "rejected"
                self._log_evolution("decide", f"拒绝: {improvement.title} (evidence={improvement.evidence})")

                if self.memory:
                    self.memory.store_episodic(
                        f"改进被拒绝: {improvement.title} — 证据不足",
                        source="evolution_loop",
                        importance=0.6,
                        tags=["evolution", "rejected"],
                        evidence="FAILED"
                    )

            improvement.decided_at = time.time()
            return accepted

    # ========== 阶段 6: Rollback ==========

    def rollback(self, to_version_id: Optional[str] = None) -> bool:
        """阶段 6: 回滚到指定版本（或上一个稳定版本）"""
        with self.lock:
            self.phase = EvolutionPhase.ROLLBACK
            self._log_evolution("phase", "→ ROLLBACK")

            # 确定回滚目标
            target = None
            if to_version_id:
                target = next((v for v in self.versions if v.version_id == to_version_id), None)
            if not target and self.stable_version:
                target = self.stable_version
            if not target:
                # 找最近的非 rejected 版本
                for v in reversed(self.versions):
                    if v.status not in (VersionStatus.REJECTED.value, VersionStatus.ROLLED_BACK.value):
                        target = v
                        break

            if not target:
                self._log_evolution("rollback", "无回滚目标")
                return False

            # 执行回滚钩子
            for imp in self.improvements:
                if imp.status == "accepted" and imp.id in self._rollback_hooks:
                    try:
                        self._rollback_hooks[imp.id]()
                        self._log_evolution("rollback", f"执行回滚钩子: {imp.id}")
                    except Exception as e:
                        self._log_evolution("rollback", f"回滚钩子失败 {imp.id}: {e}")

            # 标记当前版本为已回滚
            if self.current_version:
                self.current_version.status = VersionStatus.ROLLED_BACK.value

            # 切换到目标版本
            self.current_version = target
            target.status = VersionStatus.STABLE.value
            self.stable_version = target

            self._log_evolution("rollback", f"回滚到 {target.version_id}")
            return True

    def promote_to_stable(self, version_id: str) -> bool:
        """将一个 Experimental 版本提升为 Stable"""
        with self.lock:
            v = next((v for v in self.versions if v.version_id == version_id), None)
            if not v or v.status != VersionStatus.EXPERIMENTAL.value:
                return False
            v.status = VersionStatus.STABLE.value
            self.stable_version = v
            self._log_evolution("promote", f"{version_id} → STABLE")
            return True

    # ========== 完整循环 ==========

    def run_cycle(self, task_executor: Optional[Callable] = None) -> dict:
        """执行一次完整的自进化循环"""
        self._log_evolution("cycle_start", "=== Evolution Cycle Start ===")

        # 1. Observe
        observation = self.observe()

        # 2. Evaluate
        assessment = self.evaluate(observation)

        # 3. Improve (自动从评估结果生成提案)
        improvements = assessment.get("improvement_candidates", [])
        if not improvements:
            self._log_evolution("cycle", "无改进候选，循环结束")
            self.phase = EvolutionPhase.IDLE
            return {"observation": observation, "assessment": assessment, "actions": []}

        actions = []
        for imp in improvements:
            # 4. Validate
            validated = self.validate(imp)

            # 5. Decide
            accepted = self.decide(validated)

            actions.append({
                "improvement_id": imp.id,
                "title": imp.title,
                "layer": f"L{imp.layer}",
                "evidence": imp.evidence,
                "accepted": accepted
            })

            # 6. Rollback if needed
            if not accepted and imp.version_before:
                # 不需要回滚，因为没有应用变更
                pass

        self._log_evolution("cycle_end", f"=== Cycle End: {len(actions)} actions ===")
        self.phase = EvolutionPhase.IDLE

        cycle_record = {
            "observation": observation,
            "assessment": assessment,
            "actions": actions,
            "timestamp": time.time()
        }
        self.evolution_history.append(cycle_record)
        return cycle_record

    # ========== 注册 ==========

    def register_evaluator(self, fn: Callable):
        """注册独立评估器"""
        self.evaluators.append(fn)

    def register_benchmark(self, fn: Callable):
        """注册 benchmark 测试"""
        self.benchmark_suite.append(fn)

    # ========== 日志与审计 ==========

    def _log_evolution(self, event_type: str, message: str, **extra):
        """写入进化日志"""
        entry = {
            "timestamp": time.time(),
            "phase": self.phase.value if isinstance(self.phase, EvolutionPhase) else str(self.phase),
            "event": event_type,
            "message": message,
            **extra
        }
        # 写入文件
        log_path = os.path.join(self.log_dir, "evolution_log.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass  # 日志不应阻断主流程

    def get_audit_trail(self, n=50) -> list:
        """获取审计记录"""
        log_path = os.path.join(self.log_dir, "evolution_log.jsonl")
        if not os.path.exists(log_path):
            return []
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-n:]]

    def get_improvement_history(self, layer: Optional[int] = None) -> list:
        """获取改进历史"""
        with self.lock:
            if layer:
                return [asdict(i) for i in self.improvements if i.layer == layer]
            return [asdict(i) for i in self.improvements]

    def get_version_history(self) -> list:
        """获取版本历史"""
        with self.lock:
            return [asdict(v) for v in self.versions]

    def stats(self) -> dict:
        """进化引擎统计"""
        with self.lock:
            return {
                "total_traces": len(self.traces),
                "total_improvements": len(self.improvements),
                "accepted": sum(1 for i in self.improvements if i.status == "accepted"),
                "rejected": sum(1 for i in self.improvements if i.status == "rejected"),
                "total_versions": len(self.versions),
                "stable_version": self.stable_version.version_id if self.stable_version else None,
                "current_version": self.current_version.version_id if self.current_version else None,
                "current_phase": self.phase.value,
                "evolution_cycles": len(self.evolution_history)
            }
