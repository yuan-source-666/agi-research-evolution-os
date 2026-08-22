"""
AGI Phase 1 — 远程 DCU 验证脚本
将三个核心模块写到远程实例并运行联合集成测试
"""
import os, sys, time, json, tempfile, shutil, base64, traceback

BASE_DIR = "/root/private_data/agi_phase1"
os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

PASS = 0
FAIL = 0
ERRORS = []

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        ERRORS.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} - {detail}")

# ========== 1. 测试 Memory System ==========
print("\n=== 1. Memory System 基础测试 ===")
try:
    from memory_system import MemorySystem, SimpleEmbedding
    
    sp = os.path.join(BASE_DIR, "test_mem")
    if os.path.exists(sp):
        shutil.rmtree(sp)
    
    mem = MemorySystem(sp)
    
    # 存储
    wid = mem.store_working("当前任务：研究AGI架构", importance=0.8)
    check("工作记忆存储", len(mem.get_working()) == 1)
    
    eid = mem.store_episodic("DreamerV3世界模型在强化学习中应用", source="research", importance=0.8, tags=["世界模型","RL"])
    check("情景记忆存储", eid is not None)
    
    eid2 = mem.store_episodic("JEPA通过隐空间预测世界建模", source="research", importance=0.7, tags=["世界模型","JEPA"])
    eid3 = mem.store_episodic("持续学习面临灾难性遗忘", source="research", importance=0.9, tags=["持续学习"])
    
    # 检索
    results = mem.retrieve("世界模型", top_k=2)
    check("语义检索有结果", len(results) > 0)
    if results:
        check("检索相关性", "世界模型" in results[0][1].content or "Dreamer" in results[0][1].content)
    
    # 固化
    eid4 = mem.store_episodic("DreamerV3世界模型通关Minecraft钻石收集", importance=0.8, tags=["世界模型","RL"])
    eid5 = mem.store_episodic("DreamerV3世界模型通关八个领域任务", importance=0.8, tags=["世界模型","RL"])
    
    result = mem.consolidate(similarity_threshold=0.15, min_episodes=2)
    check("固化执行", result["consolidated"] >= 1, f"result: {result}")
    
    s = mem.stats()
    check("统计正确", s["episodic"] >= 5 and s["semantic"] >= 1, f"stats: {s}")
    
    # 冲突检测
    mem.store_semantic("持续学习已解决遗忘", importance=0.8, evidence="VERIFIED")
    mem.store_semantic("持续学习仍有遗忘问题", importance=0.8, evidence="FAILED")
    conflicts = mem.resolve_conflicts(similarity_threshold=0.2)
    check("冲突检测", len(conflicts) > 0, f"conflicts: {conflicts}")
    
    # 重新统计（包含冲突检测新增的语义记忆）
    s_final = mem.stats()
    
    # 持久化
    mem2 = MemorySystem(sp)
    s2 = mem2.stats()
    check("持久化验证", s2["episodic"] == s_final["episodic"] and s2["semantic"] == s_final["semantic"], 
          f"before={s_final}, after={s2}")
    
    print(f"  Memory System: PASS")
except Exception as e:
    check("Memory System 无异常", False, f"{e}\n{traceback.format_exc()}")

# ========== 2. 测试 Tool System ==========
print("\n=== 2. Tool System 基础测试 ===")
try:
    from tool_system import ToolSystem
    
    sp2 = os.path.join(BASE_DIR, "test_tool_mem")
    if os.path.exists(sp2):
        shutil.rmtree(sp2)
    mem_tool = MemorySystem(sp2)
    ts = ToolSystem(memory=mem_tool)
    
    tools = ts.discover()
    check("内置工具注册", len(tools) >= 7, f"got {len(tools)}")
    
    tool_names = [t["name"] for t in tools]
    check("含code_exec", "code_exec" in tool_names)
    check("含memory_store", "memory_store" in tool_names)
    
    # 代码执行
    result = ts.execute("code_exec", {"code": "result = 42\nprint('DCU test')"})
    check("代码执行", result.get("result") == "42", f"got {result}")
    check("stdout捕获", "DCU test" in result.get("stdout", ""))
    
    # Shell 执行
    result2 = ts.execute("shell_exec", {"command": "echo AGI_DCU"})
    check("Shell执行", result2.get("returncode") == 0)
    check("Shell输出", "AGI_DCU" in result2.get("stdout", ""))
    
    # 文件操作
    test_path = os.path.join(BASE_DIR, "test_file.txt")
    result3 = ts.execute("file_write", {"path": test_path, "content": "Hello from DCU!"})
    check("文件写入", result3.get("status") == "ok")
    
    result4 = ts.execute("file_read", {"path": test_path})
    check("文件读取", result4.get("content") == "Hello from DCU!")
    
    # 错误处理
    result5 = ts.execute("nonexistent", {})
    check("不存在工具错误处理", "error" in result5)
    
    result6 = ts.execute("code_exec", {"code": "1/0"})
    check("代码错误处理", "error" in result6 and "ZeroDivision" in result6.get("type", ""))
    
    # Memory 集成
    result7 = ts.execute("memory_store", {"content": "DCU测试记忆", "importance": 0.8})
    check("记忆存储工具", result7.get("status") == "ok")
    
    result8 = ts.execute("memory_retrieve", {"query": "DCU测试", "top_k": 3})
    check("记忆检索工具", len(result8.get("results", [])) > 0)
    
    # 统计
    stats = ts.stats()
    check("工具统计", stats["total_calls"] >= 7, f"got {stats}")
    
    # Memory 自动记录
    working = mem_tool.get_working()
    check("工具调用自动记录到Memory", len(working) > 0)
    
    print(f"  Tool System: PASS")
except Exception as e:
    check("Tool System 无异常", False, f"{e}\n{traceback.format_exc()}")

# ========== 3. 测试 Evolution Loop ==========
print("\n=== 3. Evolution Loop 基础测试 ===")
try:
    from evolution_loop import (
        EvolutionLoop, TaskTrace, Improvement, EvolutionPhase, 
        ImprovementLayer, EvidenceLevel
    )
    
    sp3 = os.path.join(BASE_DIR, "test_evo_mem")
    if os.path.exists(sp3):
        shutil.rmtree(sp3)
    mem_evo = MemorySystem(sp3)
    ts_evo = ToolSystem(memory=mem_evo)
    evo = EvolutionLoop(memory=mem_evo, tool_system=ts_evo, log_dir=os.path.join(BASE_DIR, "evo_logs"))
    
    # 初始化
    check("初始版本", evo.current_version is not None)
    check("phase=IDLE", evo.phase == EvolutionPhase.IDLE)
    
    # 记录轨迹
    for i in range(15):
        evo.record_trace(TaskTrace(
            task_id=f"t{i}",
            description=f"DCU任务{i}",
            start_time=time.time(),
            end_time=time.time() + 0.5,
            success=(i >= 12)  # 20% 成功率
        ))
    check("轨迹记录", len(evo.traces) == 15)
    
    # 完整循环
    result = evo.run_cycle()
    check("循环完成", isinstance(result, dict))
    check("含observation", "observation" in result)
    check("含assessment", "assessment" in result)
    check("检测到弱项", len(result["assessment"]["weaknesses"]) > 0)
    check("生成了改进候选", len(result["assessment"]["improvement_candidates"]) > 0)
    
    # 手动提案→验证→接受
    imp = evo.propose_improvement(
        layer=ImprovementLayer.L6_TOOL.value,
        title="增加工具错误重试",
        description="工具调用失败时自动重试3次",
        rationale="工具错误率高",
        expected_gain="错误率降低50%",
        risk_level="low"
    )
    check("提案创建", imp.id != "")
    
    def sandbox():
        return {"status": "ok", "tests": 5, "passed": 5}
    def benchmark():
        return {"error_rate": 0.05, "improvement": "60%", "pass": True}
    def evaluator():
        return {"overall": True, "note": "改进有效"}
    
    validated = evo.validate(imp, sandbox, benchmark, evaluator)
    check("验证通过", validated.evidence == "VERIFIED")
    
    accepted = evo.decide(validated)
    check("改进被接受", accepted == True)
    check("新版本创建", len(evo.versions) >= 2, f"got {len(evo.versions)}")
    
    # 版本历史
    versions = evo.get_version_history()
    check("版本历史", len(versions) >= 2, f"got {len(versions)}")
    
    # 回滚
    initial_vid = versions[0]["version_id"]
    rb = evo.rollback(to_version_id=initial_vid)
    check("回滚成功", rb == True)
    check("回滚到原版本", evo.current_version.version_id == initial_vid)
    
    # 审计
    audit = evo.get_audit_trail(n=20)
    check("审计记录", len(audit) > 0)
    
    # 统计
    stats = evo.stats()
    check("进化统计", stats["total_traces"] == 15)
    check("改进数", stats["total_improvements"] >= 1)
    
    print(f"  Evolution Loop: PASS")
except Exception as e:
    check("Evolution Loop 无异常", False, f"{e}\n{traceback.format_exc()}")

# ========== 4. 联合集成测试 ==========
print("\n=== 4. 联合集成测试 ===")
try:
    # 三模块联合：Memory + Tool + Evolution 完整流程
    sp4 = os.path.join(BASE_DIR, "test_integration")
    if os.path.exists(sp4):
        shutil.rmtree(sp4)
    
    mem_int = MemorySystem(sp4)
    ts_int = ToolSystem(memory=mem_int)
    evo_int = EvolutionLoop(memory=mem_int, tool_system=ts_int, log_dir=os.path.join(sp4, "logs"))
    
    # 模拟任务执行
    for i in range(10):
        # 用工具执行任务
        tool_result = ts_int.execute("code_exec", {"code": f"result = {i} * 2"})
        success = "error" not in tool_result
        
        evo_int.record_trace(TaskTrace(
            task_id=f"int_{i}",
            description=f"集成测试任务{i}",
            start_time=time.time(),
            end_time=time.time() + 0.1,
            tools_used=["code_exec"],
            success=success
        ))
    
    # 运行进化循环
    cycle = evo_int.run_cycle()
    check("联合循环完成", "observation" in cycle)
    
    # 工具调用记录在 Memory 中
    all_mem = mem_int.get_all()
    check("Memory含工具调用记录", len(all_mem) > 0)
    
    # 检索进化相关记忆
    evo_results = mem_int.retrieve("改进", top_k=5)
    check("Memory含进化记录", len(evo_results) > 0)
    
    # 进化日志写入文件
    log_path = os.path.join(sp4, "logs", "evolution_log.jsonl")
    check("进化日志文件存在", os.path.exists(log_path))
    
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_lines = f.readlines()
        check("日志非空", len(log_lines) > 0)
        # 验证日志是有效JSON
        first_entry = json.loads(log_lines[0])
        check("日志有效JSON", "timestamp" in first_entry and "event" in first_entry)
    
    print(f"  联合集成: PASS")
except Exception as e:
    check("联合集成无异常", False, f"{e}\n{traceback.format_exc()}")

# ========== 结果 ==========
print("\n" + "=" * 60)
print(f"远程 DCU 验证结果: {PASS} PASS / {FAIL} FAIL")
if ERRORS:
    print("\n失败详情:")
    for e in ERRORS:
        print(f"  - {e}")
print("=" * 60)
print(f"环境: Python {sys.version.split()[0]}, 平台 {sys.platform}")
print(f"路径: {BASE_DIR}")
