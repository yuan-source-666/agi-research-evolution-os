#!/usr/bin/env python3
"""
AGI Core Orchestrator v0.5
==========================
Integrates inference engine, memory, tools, evolution, benchmark, continual learning, and world model.

Architecture:
  - Inference Engine (agi_engine.py :8080) -- Qwen2.5-32B-Instruct, cognitive core
  - Memory System (memory_system.py) -- hierarchical persistent memory
  - Tool System (tool_system.py) -- standardized tool execution
  - Evolution Loop (evolution_loop.py) -- self-evolution engine
  - Benchmark System (benchmark_system.py) -- standardized evaluation suite
  - Continual Learning (continual_learning.py) -- experience replay + skill extraction
  - World Model (world_model.py) -- latent-space state prediction + imagination
  - agi_core.py (:9099) -- external API + Agent Loop orchestration

Agent Loop:
  1. Receive user input
  2. Retrieve relevant memories + relevant skills
  3. Send user input + memories + tool descriptions to inference engine
  4. Inference engine generates response (may include tool calls)
  5. If tool calls -> execute tools -> feed results back to engine -> goto 4
  6. Inference engine gives final response
  7. Store interaction to memory system
  8. Record execution trace to evolution loop
  9. Record experience to continual learning
  10. Return final response to user

Dependencies: requests, fastapi, uvicorn, pydantic + local modules
"""

import os, sys, json, time, uuid, traceback, asyncio, threading
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dataclasses import asdict
import uvicorn

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_system import MemorySystem
from tool_system import ToolSystem
from evolution_loop import EvolutionLoop, TaskTrace
from benchmark_system import TaskRunner, TaskSuite, run_benchmark
from continual_learning import ContinualLearning
from world_model import WorldModel

# ============================================================
# Configuration
# ============================================================
INFERENCE_URL = os.environ.get("AGI_INFERENCE_URL", "http://127.0.0.1:8080")
CORE_PORT = int(os.environ.get("AGI_CORE_PORT", "9099"))
DATA_DIR = os.environ.get("AGI_DATA_DIR", "/root/private_data/agi_data")
MAX_TOOL_ROUNDS = 5  # Max tool call rounds to prevent infinite loops
INFERENCE_TIMEOUT = 300  # Inference engine timeout (seconds) - 32B model needs more time
INFERENCE_TIMEOUT_LIGHT = 120  # Light inference (no tools, fewer tokens)

os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# Initialize subsystems
# ============================================================
print("=" * 60, flush=True)
print("[AGI Core] Initializing AGI Orchestrator v0.5", flush=True)
print(f"[AGI Core] Inference engine: {INFERENCE_URL}", flush=True)
print(f"[AGI Core] Data directory: {DATA_DIR}", flush=True)
print(f"[AGI Core] Core port: {CORE_PORT}", flush=True)
print("=" * 60, flush=True)

# 1. Memory System
print("[1/7] Loading Memory System...", flush=True)
memory = MemorySystem(os.path.join(DATA_DIR, "memory"))
print(f"  Memory stats: {memory.stats()}", flush=True)

# 2. Tool System
print("[2/7] Loading Tool System...", flush=True)
tools = ToolSystem(memory=memory)
print(f"  Tools registered: {len(tools.tools)} ({', '.join(tools.tools.keys())})", flush=True)

# 3. Evolution Loop
print("[3/7] Loading Evolution Loop...", flush=True)
evolution = EvolutionLoop(
    memory=memory,
    tool_system=tools,
    log_dir=os.path.join(DATA_DIR, "evolution_logs")
)
print(f"  Evolution version: {evolution.current_version.version_id}", flush=True)

# 4. Benchmark System
print("[4/7] Loading Benchmark System...", flush=True)
benchmark_runner = TaskRunner(agi_core_url=f"http://127.0.0.1:{CORE_PORT}")
print(f"  Benchmark tasks: {benchmark_runner.suite.count()}", flush=True)

# 5. Continual Learning
print("[5/7] Loading Continual Learning...", flush=True)
def _run_quick_benchmark():
    """Quick benchmark for regression check (subset of tasks)."""
    try:
        report = benchmark_runner.run_suite(
            task_ids=["reason_001", "tool_001", "code_001"],
            version_id=evolution.current_version.version_id if evolution.current_version else ""
        )
        from dataclasses import asdict
        return asdict(report)
    except Exception as e:
        return {"overall_score": 0.0, "error": str(e)[:200]}

continual_learning = ContinualLearning(
    memory=memory,
    evolution=evolution,
    benchmark_fn=_run_quick_benchmark,
    log_dir=os.path.join(DATA_DIR, "cl_logs")
)
print(f"  Skill library: {continual_learning.stats()['skill_library_size']} skills", flush=True)

# 6. World Model
print("[6/7] Loading World Model...", flush=True)
world_model = WorldModel(
    inference_url=INFERENCE_URL,
    inference_timeout=INFERENCE_TIMEOUT_LIGHT,
    log_dir=os.path.join(DATA_DIR, "wm_logs")
)
print(f"  World model states: {world_model.stats()['total_states']}", flush=True)

# 7. Inference engine connection check
print("[7/7] Checking inference engine...", flush=True)
_engine_ok = False
for attempt in range(3):
    try:
        resp = requests.get(f"{INFERENCE_URL}/health", timeout=10)
        if resp.status_code == 200:
            health = resp.json()
            print(f"  Engine health: {health}", flush=True)
            _engine_ok = True
            break
    except Exception as e:
        print(f"  Attempt {attempt+1}/3 failed: {e}", flush=True)
        time.sleep(2)

if not _engine_ok:
    print("  [WARNING] Inference engine not reachable! API will start but inference will fail.", flush=True)
    print("  [WARNING] Make sure agi_engine.py is running on port 8080.", flush=True)

print("=" * 60, flush=True)
print("[AGI Core] All 7 systems initialized.", flush=True)

# ============================================================
# Inference Engine Interface
# ============================================================
def call_inference(messages, tools_schema=None, max_tokens=1024, temperature=0.7, top_p=0.9):
    """Call inference engine (agi_engine.py :8080)"""
    payload = {
        "model": "qwen2.5-32b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if tools_schema:
        payload["tools"] = tools_schema

    resp = requests.post(
        f"{INFERENCE_URL}/v1/chat/completions",
        json=payload,
        timeout=INFERENCE_TIMEOUT
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Inference engine error: {resp.status_code} {resp.text}")

    data = resp.json()
    choice = data["choices"][0]
    msg = choice["message"]
    gen_time = data.get("timings", {}).get("generation_time_s", 0)

    return {
        "content": msg.get("content", ""),
        "tool_calls": msg.get("tool_calls"),
        "finish_reason": choice.get("finish_reason", "stop"),
        "gen_time": gen_time,
    }

def call_inference_light(messages, max_tokens=512, temperature=0.7, top_p=0.9):
    """Lightweight inference call -- no tools, fewer tokens, shorter timeout.
    Used for planning and synthesis steps in research cycle to avoid timeout."""
    payload = {
        "model": "qwen2.5-32b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    resp = requests.post(
        f"{INFERENCE_URL}/v1/chat/completions",
        json=payload,
        timeout=INFERENCE_TIMEOUT_LIGHT
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Inference engine error: {resp.status_code} {resp.text}")
    data = resp.json()
    choice = data["choices"][0]
    msg = choice["message"]
    gen_time = data.get("timings", {}).get("generation_time_s", 0)
    return {
        "content": msg.get("content", ""),
        "tool_calls": None,
        "finish_reason": choice.get("finish_reason", "stop"),
        "gen_time": gen_time,
    }

# ============================================================
# Agent Loop Core Logic
# ============================================================
def build_tool_schema():
    """Build OpenAI-format tool descriptions from ToolSystem"""
    tool_list = tools.discover()
    result = []
    for t in tool_list:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["schema"],
            }
        })
    return result

def build_system_prompt(user_input, retrieved_memories):
    """Build system prompt, inject memory and identity"""
    parts = [
        "You are an AGI (Artificial General Intelligence) system running on a remote DCU platform.",
        "You have persistent memory, tools, and a self-evolution engine.",
        "You can use tools to accomplish tasks. When you need to use a tool, output a tool call.",
        "After tool results are provided, continue your response naturally.",
        "",
        "## Your Capabilities",
        f"- Memory System: {memory.stats()}",
        f"- Tool System: {len(tools.tools)} tools available",
        f"- Evolution Engine: version {evolution.current_version.version_id}",
        "",
    ]

    if retrieved_memories:
        parts.append("## Relevant Memories")
        for score, item in retrieved_memories:
            parts.append(f"- [{item.type}] (score={score:.3f}, evidence={item.evidence}) {item.content[:200]}")
        parts.append("")
    else:
        parts.append("## Relevant Memories")
        parts.append("(No relevant memories found for this query.)")
        parts.append("")

    parts.append("## Instructions")
    parts.append("- Think step by step before acting.")
    parts.append("- Use tools when you need information or computation.")
    parts.append("- Be concise and accurate.")
    parts.append("- Respond in the user's language.")

    # Inject relevant skills from continual learning
    relevant_skills = continual_learning.find_relevant_skills(user_input, top_k=3)
    if relevant_skills:
        parts.append("")
        parts.append("## Relevant Skills")
        for skill in relevant_skills:
            parts.append(f"- [{skill.type}] {skill.name}: {skill.description[:150]}")

    return "\n".join(parts)

def execute_tool_call(tool_call):
    """Execute a tool call returned by the inference engine"""
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    args_str = fn.get("arguments", "{}")

    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except json.JSONDecodeError:
        args = {"raw": args_str}

    print(f"  [Tool Call] {name}({json.dumps(args, ensure_ascii=False)[:200]})", flush=True)

    result = tools.execute(name, args)

    # Truncate overly long results
    result_str = json.dumps(result, ensure_ascii=False, default=str)
    if len(result_str) > 4000:
        result_str = result_str[:4000] + "...[truncated]"

    print(f"  [Tool Result] {result_str[:200]}", flush=True)

    return {
        "name": name,
        "result": result_str,
        "call_id": tool_call.get("id", ""),
    }

def agent_loop(user_input, conversation_history=None, max_rounds=MAX_TOOL_ROUNDS):
    """
    Core Agent Loop:
      retrieve memories -> infer -> (tool call -> infer)* -> final response -> store memory -> record trace
    Returns: {response, tool_calls_log, memories_used, gen_time, rounds}
    """
    t_start = time.time()

    # 1. Retrieve relevant memories from memory system
    retrieved = memory.retrieve(user_input, top_k=5)
    retrieved_contents = [item.content for _, item in retrieved]

    # 2. Build system prompt
    sys_prompt = build_system_prompt(user_input, retrieved)

    # 3. Build messages
    messages = []
    messages.append({"role": "system", "content": sys_prompt})
    if conversation_history:
        for msg in conversation_history[-10:]:  # Keep last 10 turns
            messages.append(msg)
    messages.append({"role": "user", "content": user_input})

    # 4. Tool descriptions
    tool_schema = build_tool_schema()

    # 5. Agent Loop
    all_tool_calls = []
    round_num = 0

    while round_num < max_rounds:
        round_num += 1
        print(f"[AGI Core] Agent loop round {round_num}/{max_rounds}", flush=True)

        result = call_inference(
            messages=messages,
            tools_schema=tool_schema if round_num == 1 else tool_schema,
            max_tokens=1024,
            temperature=0.7,
        )

        content = result["content"] or ""
        tool_calls = result["tool_calls"]
        gen_time = result["gen_time"]

        if not tool_calls:
            # No tool calls -> final response
            # Store to memory
            memory.store_episodic(
                content=f"User: {user_input}\nAssistant: {content}",
                source="conversation",
                importance=0.6,
                tags=["conversation", "response"],
                evidence="VERIFIED"
            )

            total_time = time.time() - t_start

            # Record TaskTrace to evolution engine
            trace = TaskTrace(
                task_id=f"chat_{uuid.uuid4().hex[:8]}",
                description=user_input[:200],
                start_time=t_start,
                end_time=time.time(),
                tools_used=[tc["name"] for tc in all_tool_calls],
                steps=[
                    {"action": tc["name"], "result": tc["result_preview"][:100], "success": True}
                    for tc in all_tool_calls
                ],
                success=True,
                metrics={
                    "rounds": round_num,
                    "memories_used": len(retrieved),
                    "gen_time": gen_time,
                    "total_time": total_time,
                    "tool_calls": len(all_tool_calls),
                },
            )
            evolution.record_trace(trace)

            # Record experience to continual learning
            try:
                continual_learning.record_experience(
                    trace_id=trace.task_id,
                    task_description=user_input,
                    tools_used=[tc["name"] for tc in all_tool_calls],
                    steps=all_tool_calls,
                    success=True,
                    metrics={"rounds": round_num, "total_time": total_time, "score": 1.0}
                )
            except Exception as e:
                print(f"  [ContinualLearning] Record failed: {e}", flush=True)

            # Register interaction as a world state
            try:
                world_model.register_state(
                    description=f"User asked: {user_input[:100]} -> Response given",
                    properties={"success": True, "tools": len(all_tool_calls)}
                )
            except Exception:
                pass

            return {
                "response": content,
                "tool_calls_log": all_tool_calls,
                "memories_used": len(retrieved),
                "gen_time_total": gen_time,
                "total_time": round(total_time, 3),
                "rounds": round_num,
            }

        # Has tool calls -> execute
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            tool_result = execute_tool_call(tc)
            all_tool_calls.append({
                "name": tool_result["name"],
                "arguments": tc.get("function", {}).get("arguments", ""),
                "result_preview": tool_result["result"][:200],
                "call_id": tool_result["call_id"],
            })

            # Feed tool results back to inference engine
            messages.append({
                "role": "tool",
                "name": tool_result["name"],
                "content": tool_result["result"],
                "tool_call_id": tool_result["call_id"],
            })

        # Continue loop, let inference engine process tool results

    # Exceeded max rounds - record trace with failure
    total_time = time.time() - t_start

    trace = TaskTrace(
        task_id=f"chat_{uuid.uuid4().hex[:8]}",
        description=user_input[:200],
        start_time=t_start,
        end_time=time.time(),
        tools_used=[tc["name"] for tc in all_tool_calls],
        steps=[
            {"action": tc["name"], "result": tc["result_preview"][:100], "success": True}
            for tc in all_tool_calls
        ],
        success=False,
        error="max_rounds_reached",
        metrics={
            "rounds": max_rounds,
            "memories_used": len(retrieved),
            "total_time": total_time,
            "tool_calls": len(all_tool_calls),
        },
    )
    evolution.record_trace(trace)

    # Record failed experience to continual learning
    try:
        continual_learning.record_experience(
            trace_id=trace.task_id,
            task_description=user_input,
            tools_used=[tc["name"] for tc in all_tool_calls],
            steps=all_tool_calls,
            success=False,
            metrics={"rounds": max_rounds, "total_time": total_time, "score": 0.0}
        )
    except Exception:
        pass

    return {
        "response": "I've reached the maximum number of tool call rounds. Here's what I found so far: " + (content or ""),
        "tool_calls_log": all_tool_calls,
        "memories_used": len(retrieved),
        "gen_time_total": 0,
        "total_time": round(total_time, 2),
        "rounds": max_rounds,
        "warning": "max_rounds_reached",
    }

# ============================================================
# Autonomous Research Mode
# ============================================================
def autonomous_research_cycle(topic=None):
    """
    AGI Autonomous Research Loop (optimized v2):
      1. Generate research plan (light inference -- no tools)
      2. Execute research steps (agent_loop with tools, reduced rounds)
      3. Store findings to memory
      4. Synthesize conclusions (light inference -- no tools)
      5. Record evolution trace
      6. Run evolution cycle
    Progress is tracked via _research_state for real-time monitoring.
    """
    print(f"[AGI Core] Starting autonomous research cycle v2", flush=True)
    cycle_start = time.time()
    total_steps = 6  # plan + 3 steps + synthesis + evolution
    _research_state["current_step"] = 0
    _research_state["total_steps"] = total_steps
    _research_state["step_desc"] = "Initializing"

    def update_progress(step, desc):
        _research_state["current_step"] = step
        _research_state["step_desc"] = desc
        print(f"[AGI Core] Research step {step}/{total_steps}: {desc}", flush=True)

    # If no topic specified, find recent interests from memory
    if not topic:
        recent = memory.get_all("episodic", limit=5)
        if recent:
            topic = "Analyze and synthesize recent experiences: " + "; ".join(r.content[:100] for r in recent)
        else:
            topic = "What are the most important open problems in AGI research today?"

    # 1. Generate research plan (lightweight -- no tools, 512 tokens)
    update_progress(1, "Generating research plan")
    plan_messages = [
        {"role": "system", "content": "You are a research planner. Be concise. Output 2-3 numbered steps only."},
        {"role": "user", "content": f"Design a brief research plan to investigate: {topic}\nList 2-3 concrete steps. Be very concise."}
    ]
    try:
        plan_result = call_inference_light(plan_messages, max_tokens=512)
        plan = plan_result["content"] or "1. Analyze the topic 2. Search for information 3. Summarize findings"
    except Exception as e:
        print(f"[AGI Core] Plan generation failed: {e}", flush=True)
        plan = "1. Analyze the topic 2. Search for information 3. Summarize findings"

    # 2. Execute research steps
    findings = []
    steps = [s.strip() for s in plan.split("\n") if s.strip() and s.strip()[0].isdigit()]
    if not steps:
        steps = ["1. Analyze the topic", "2. Search for information", "3. Summarize findings"]

    all_tools_used = []
    for i, step in enumerate(steps[:3]):
        update_progress(2 + i, f"Executing step {i+1}: {step[:80]}")
        step_prompt = f"Execute this research step concisely: {step}\nUse available tools if needed. Report findings in 2-3 sentences."
        try:
            step_result = agent_loop(step_prompt, max_rounds=3)
            findings.append(step_result["response"])
            all_tools_used.extend(tc["name"] for tc in step_result.get("tool_calls_log", []))
        except Exception as e:
            print(f"[AGI Core] Step {i+1} failed: {e}", flush=True)
            findings.append(f"[Step {i+1} failed: {str(e)[:100]}]")

        # Store findings
        try:
            memory.store_episodic(
                content=f"Research finding [{i+1}]: {findings[-1][:500]}",
                source="autonomous_research",
                importance=0.8,
                tags=["research", "autonomous", f"step_{i+1}"],
                evidence="VERIFIED"
            )
        except Exception as e:
            print(f"[AGI Core] Memory store failed for step {i+1}: {e}", flush=True)

    # 4. Synthesize conclusions (lightweight -- no tools, 512 tokens)
    update_progress(5, "Synthesizing conclusions")
    synthesis_messages = [
        {"role": "system", "content": "You are a research synthesizer. Be concise. Provide key insights and next steps."},
        {"role": "user", "content": (
            f"Synthesize these research findings into a concise summary:\n\n"
            + "\n\n".join(f"Finding {i+1}: {f[:300]}" for i, f in enumerate(findings))
            + "\n\nProvide key insights and next steps in 3-5 sentences."
        )}
    ]
    try:
        synthesis_result = call_inference_light(synthesis_messages, max_tokens=512)
        synthesis = synthesis_result["content"] or "Synthesis unavailable."
    except Exception as e:
        print(f"[AGI Core] Synthesis failed: {e}", flush=True)
        synthesis = f"Synthesis failed: {str(e)[:100]}"

    # 5. Store synthesis to semantic memory
    try:
        memory.store_semantic(
            content=f"Research synthesis on '{topic[:200]}': {synthesis[:500]}",
            source="autonomous_research",
            importance=0.9,
            tags=["research", "synthesis", "autonomous"],
            evidence="VERIFIED"
        )
    except Exception as e:
        print(f"[AGI Core] Semantic memory store failed: {e}", flush=True)

    # 6. Record execution trace to evolution loop
    trace = TaskTrace(
        task_id=f"auto_{uuid.uuid4().hex[:8]}",
        description=f"Autonomous research: {topic[:100]}",
        start_time=cycle_start,
        end_time=time.time(),
        tools_used=list(set(all_tools_used)),
        steps=[{"finding": f[:200]} for f in findings],
        success=True,
        metrics={"steps": len(findings), "synthesis_length": len(synthesis)},
    )
    evolution.record_trace(trace)

    # 7. Run evolution cycle
    update_progress(6, "Running evolution cycle")
    try:
        cycle_result = evolution.run_cycle()
    except Exception as e:
        print(f"[AGI Core] Evolution cycle failed: {e}", flush=True)
        cycle_result = {"assessment": {"overall_score": 0, "weaknesses": []}, "actions": []}

    cycle_time = time.time() - cycle_start
    print(f"[AGI Core] Research cycle complete in {cycle_time:.1f}s", flush=True)

    return {
        "topic": topic,
        "plan": plan[:500],
        "findings": findings,
        "synthesis": synthesis,
        "cycle_time": round(cycle_time, 2),
        "evolution": {
            "overall_score": cycle_result.get("assessment", {}).get("overall_score", 0),
            "weaknesses": len(cycle_result.get("assessment", {}).get("weaknesses", [])),
            "actions": cycle_result.get("actions", []),
        },
        "trace_id": trace.task_id,
    }

# ============================================================
# FastAPI External Interface
# ============================================================
app = FastAPI(title="AGI Core Orchestrator", version="0.1.0")

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    max_rounds: Optional[int] = MAX_TOOL_ROUNDS

class ResearchRequest(BaseModel):
    topic: Optional[str] = None

@app.get("/")
async def root():
    return {
        "system": "AGI Core Orchestrator",
        "version": "0.5.0",
        "inference_engine": INFERENCE_URL,
        "endpoints": ["/chat", "/research", "/status", "/memory", "/tools", "/evolution",
                       "/benchmark", "/learning", "/worldmodel"],
    }

@app.get("/status")
async def status():
    """Overall system status"""
    engine_health = None
    try:
        resp = requests.get(f"{INFERENCE_URL}/health", timeout=5)
        if resp.status_code == 200:
            engine_health = resp.json()
    except Exception:
        engine_health = {"status": "unreachable"}

    return {
        "system": "AGI Core v0.5",
        "inference_engine": engine_health,
        "memory": memory.stats(),
        "tools": tools.stats(),
        "evolution": evolution.stats(),
        "continual_learning": continual_learning.stats(),
        "world_model": world_model.stats(),
        "benchmark": {"total_tasks": benchmark_runner.suite.count(),
                       "history_count": len(benchmark_runner.results_history)},
        "data_dir": DATA_DIR,
        "uptime": round(time.time() - _start_time, 1),
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    """Chat endpoint -- full Agent Loop"""
    try:
        history = None
        if req.history:
            history = [{"role": m.role, "content": m.content or ""} for m in req.history]

        result = agent_loop(
            user_input=req.message,
            conversation_history=history,
            max_rounds=req.max_rounds or MAX_TOOL_ROUNDS,
        )
        return JSONResponse(content=result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Async Research State
# ============================================================
_research_state = {
    "status": "idle",  # idle / running / completed / failed
    "task_id": None,
    "result": None,
    "error": None,
    "started_at": 0,
    "completed_at": 0,
    "current_step": 0,
    "total_steps": 6,
    "step_desc": "",
}

@app.post("/research")
async def research(req: ResearchRequest):
    """Autonomous research endpoint -- starts async research cycle"""
    if _research_state["status"] == "running":
        return JSONResponse(content={
            "status": "already_running",
            "task_id": _research_state["task_id"],
            "message": "A research cycle is already in progress. Use GET /research/status to check."
        })
    
    task_id = f"research_{uuid.uuid4().hex[:8]}"
    _research_state["status"] = "running"
    _research_state["task_id"] = task_id
    _research_state["result"] = None
    _research_state["error"] = None
    _research_state["started_at"] = time.time()
    _research_state["completed_at"] = 0
    
    topic = req.topic
    
    def run_research():
        try:
            result = autonomous_research_cycle(topic=topic)
            _research_state["status"] = "completed"
            _research_state["result"] = result
            _research_state["completed_at"] = time.time()
        except Exception as e:
            _research_state["status"] = "failed"
            _research_state["error"] = str(e)
            _research_state["completed_at"] = time.time()
            traceback.print_exc()
    
    thread = threading.Thread(target=run_research, daemon=True)
    thread.start()
    
    return JSONResponse(content={
        "status": "started",
        "task_id": task_id,
        "message": "Research cycle started in background. Use GET /research/status to check progress."
    })

@app.get("/research/status")
async def research_status():
    """Check status of async research cycle"""
    elapsed = 0
    if _research_state["completed_at"] > 0:
        elapsed = _research_state["completed_at"] - _research_state["started_at"]
    elif _research_state["started_at"] > 0:
        elapsed = time.time() - _research_state["started_at"]
    
    response = {
        "status": _research_state["status"],
        "task_id": _research_state["task_id"],
        "elapsed_seconds": round(elapsed, 1),
        "current_step": _research_state.get("current_step", 0),
        "total_steps": _research_state.get("total_steps", 6),
        "step_desc": _research_state.get("step_desc", ""),
    }
    
    if _research_state["status"] == "completed" and _research_state["result"]:
        result = _research_state["result"]
        response["topic"] = result.get("topic", "")[:200]
        response["plan"] = result.get("plan", "")[:300]
        response["findings_count"] = len(result.get("findings", []))
        response["synthesis"] = result.get("synthesis", "")[:500]
        response["cycle_time"] = result.get("cycle_time")
        response["evolution"] = result.get("evolution", {})
        response["trace_id"] = result.get("trace_id")
    elif _research_state["status"] == "failed":
        response["error"] = _research_state["error"]
    
    return JSONResponse(content=response)

@app.get("/memory")
async def memory_status():
    """Memory system status"""
    stats = memory.stats()
    recent = memory.get_all(limit=20)
    return {
        "stats": stats,
        "recent_memories": [
            {
                "id": m.id,
                "type": m.type,
                "content": m.content[:200],
                "importance": m.importance,
                "evidence": m.evidence,
                "timestamp": m.timestamp,
            }
            for m in recent
        ],
    }

@app.get("/memory/search")
async def memory_search(q: str, top_k: int = 5):
    """Search memories"""
    results = memory.retrieve(q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {
                "score": round(s, 4),
                "type": item.type,
                "content": item.content,
                "importance": item.importance,
                "evidence": item.evidence,
            }
            for s, item in results
        ],
    }

@app.get("/tools")
async def tools_status():
    """Tool system status"""
    return {
        "stats": tools.stats(),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "permission": t.permission,
                "calls": t.call_count,
                "avg_latency": round(t.avg_latency, 4),
                "error_rate": round(t.error_rate, 4),
            }
            for t in tools.tools.values()
        ],
        "recent_log": tools.get_log(10),
    }

@app.get("/evolution")
async def evolution_status():
    """Evolution loop status"""
    return {
        "stats": evolution.stats(),
        "version_history": evolution.get_version_history(),
        "recent_improvements": evolution.get_improvement_history()[-10:],
        "audit_trail": evolution.get_audit_trail(20),
    }

@app.post("/evolution/cycle")
async def evolution_cycle():
    """Manually trigger an evolution cycle"""
    try:
        result = evolution.run_cycle()
        # Convert Improvement dataclass objects to dicts for JSON serialization
        def safe_serialize(obj):
            if hasattr(obj, '__dict__'):
                return {k: safe_serialize(v) for k, v in vars(obj).items()}
            if isinstance(obj, list):
                return [safe_serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: safe_serialize(v) for k, v in obj.items()}
            return obj
        return JSONResponse(content=safe_serialize(result))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/consolidate")
async def memory_consolidate():
    """Trigger memory consolidation (episodic -> semantic)"""
    result = memory.consolidate()
    return result

# ============================================================
# Phase 3: Benchmark / Continual Learning / World Model Endpoints
# ============================================================

class BenchmarkRequest(BaseModel):
    categories: Optional[List[str]] = None
    task_ids: Optional[List[str]] = None

class WorldModelRequest(BaseModel):
    initial_state: str
    goal: str
    actions: List[str]
    depth: Optional[int] = 3

@app.post("/benchmark")
async def run_benchmark_endpoint(req: BenchmarkRequest = None):
    """Run benchmark suite against the AGI system."""
    try:
        version_id = evolution.current_version.version_id if evolution.current_version else ""
        # Run in background thread to avoid blocking
        _bench_state["status"] = "running"
        _bench_state["started_at"] = time.time()

        def run_bg():
            try:
                report = benchmark_runner.run_suite(
                    categories=req.categories if req else None,
                    task_ids=req.task_ids if req else None,
                    version_id=version_id
                )
                _bench_state["status"] = "completed"
                _bench_state["result"] = asdict(report)
                _bench_state["completed_at"] = time.time()
            except Exception as e:
                _bench_state["status"] = "failed"
                _bench_state["error"] = str(e)
                _bench_state["completed_at"] = time.time()

        thread = threading.Thread(target=run_bg, daemon=True)
        thread.start()

        return JSONResponse(content={
            "status": "started",
            "message": "Benchmark running in background. Use GET /benchmark/status to check.",
            "total_tasks": benchmark_runner.suite.count()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

_bench_state = {"status": "idle", "result": None, "error": None, "started_at": 0, "completed_at": 0}

@app.get("/benchmark/status")
async def benchmark_status():
    """Check benchmark status."""
    elapsed = 0
    if _bench_state["completed_at"] > 0:
        elapsed = _bench_state["completed_at"] - _bench_state["started_at"]
    elif _bench_state["started_at"] > 0:
        elapsed = time.time() - _bench_state["started_at"]

    response = {
        "status": _bench_state["status"],
        "elapsed_seconds": round(elapsed, 1),
    }
    if _bench_state["status"] == "completed" and _bench_state["result"]:
        response.update(_bench_state["result"])
    elif _bench_state["status"] == "failed":
        response["error"] = _bench_state["error"]
    return JSONResponse(content=response)

@app.get("/benchmark/history")
async def benchmark_history():
    """Get benchmark history."""
    return JSONResponse(content=benchmark_runner.get_history(10))

@app.post("/learning")
async def run_learning_cycle():
    """Run a continual learning cycle (replay + skill extraction + consolidation + regression check)."""
    try:
        report = continual_learning.run_learning_cycle()
        return JSONResponse(content=asdict(report))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/status")
async def learning_status():
    """Get continual learning status."""
    return JSONResponse(content=continual_learning.stats())

@app.get("/learning/skills")
async def learning_skills():
    """List all extracted skills."""
    stats = continual_learning.stats()
    return JSONResponse(content={
        "total": stats["skill_library_size"],
        "skills": stats.get("skills", [])
    })

@app.post("/worldmodel")
async def worldmodel_plan(req: WorldModelRequest):
    """Use world model to plan actions toward a goal."""
    try:
        result = world_model.plan_to_goal(
            initial_state=req.initial_state,
            goal=req.goal,
            possible_actions=req.actions,
            depth=req.depth or 3
        )
        return JSONResponse(content=result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/worldmodel/imagine")
async def worldmodel_imagine(req: WorldModelRequest):
    """Use world model to imagine future trajectories."""
    try:
        state = world_model.register_state(req.initial_state)
        trajectories = world_model.imagine_futures(
            state.id, req.actions, depth=req.depth or 3
        )
        return JSONResponse(content={
            "state_id": state.id,
            "trajectories": [asdict(t) for t in trajectories],
            "total": len(trajectories),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/worldmodel")
async def worldmodel_status():
    """Get world model status."""
    return JSONResponse(content=world_model.stats())

@app.get("/worldmodel/states")
async def worldmodel_states():
    """Get registered world states."""
    return JSONResponse(content=world_model.get_state_history(20))

# ============================================================
# Main
# ============================================================
_start_time = time.time()

if __name__ == "__main__":
    print(f"[AGI Core] Starting server on 0.0.0.0:{CORE_PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=CORE_PORT, log_level="info")
