# -*- coding: utf-8 -*-
"""
阶段 B 首次点火：evolution_loop 引擎第一次完整自进化闭环。

玩具任务族（可验证、确定性种子）：
- 任务：整数算术 (a op b)，共 120 个固定任务
- 工具 compute：以 P_CORRUPT 概率返回损坏值，但 checksum 始终由真值计算
  → 校验和验证可以 100% 检测损坏（改进是"真"的，不是作弊）
- 智能体配置（策略层）：
    verify  : 是否做 checksum 校验（初始 False）
    retries : 检测到损坏后的重试次数（初始 0）

理论成功率：verify=False → 0.45；verify+retry2 → 1-0.55^3≈0.83；retry4 → 1-0.55^5≈0.95

闭环剧本（全部走引擎真实 API）：
  C0 基线观测 → 弱点 task_success_rate
  ImpA L3策略 verify+retry2 → 沙箱/基准/独立评估 → 接受 → v2
  C1 观测 sr≈0.83
  ImpB L3策略 retry2→4 → 接受 → v3 → C2 sr≈0.95 → promote_to_stable
  ImpC L6工具 关闭校验（省调用）→ 基准 sr 崩 → 证据 FAILED → 拒绝
  ImpD L4流程 retry4→2（省算力）→ 基准子集阈值有缺陷 → 放行 → v4
  C3 全量新种子观测 sr≈0.83 → 检出回归 → rollback() → 配置恢复
  C4 复测 sr≈0.95 → 恢复确认；记忆固化教训；consolidate
"""
import sys, json, time, random, copy, traceback

sys.path.insert(0, '/root/private_data')
from evolution_loop import EvolutionLoop, TaskTrace, ImprovementLayer
from memory_system import MemorySystem

OUT = '/root/private_data/phase_b_first_fire.json'
P_CORRUPT = 0.55
N_TASKS = 120


# ---------- 玩具世界 ----------
def make_tasks(n, seed=7):
    r = random.Random(seed)
    ops = ['+', '-', '*']
    tasks = []
    for i in range(n):
        a, b = r.randint(10, 99), r.randint(2, 9)
        op = r.choice(ops)
        correct = a + b if op == '+' else (a - b if op == '-' else a * b)
        tasks.append({'id': f't{i:03d}', 'a': a, 'b': b, 'op': op, 'correct': correct})
    return tasks


TASKS = make_tasks(N_TASKS)


def checksum(task, value):
    return (task['a'] * 1000003 + task['b'] * 91 + ord(task['op']) * 7 + value * 13) & 0xffff


def tool_call(task, rng):
    """易错工具：值可能损坏，checksum 始终来自真值。"""
    correct = task['correct']
    corrupted = rng.random() < P_CORRUPT
    value = correct + (rng.randint(1, 9) * 10 if corrupted else 0)
    return {'value': value, 'checksum': checksum(task, correct), 'corrupted': corrupted}


def solve_one(task, config, rng):
    t0 = time.perf_counter()
    steps, calls = [], 0
    out = None
    for attempt in range(1 + config['retries']):
        out = tool_call(task, rng)
        calls += 1
        if config['verify']:
            ok = checksum(task, out['value']) == out['checksum']
        else:
            ok = True  # 不校验 → 直接采信第一次结果
        steps.append({'action': f'compute#{attempt}', 'result': out['value'],
                      'success': ok, 'latency': round(time.perf_counter() - t0, 6)})
        if ok:
            break
    return {'value': out['value'], 'correct': task['correct'],
            'success': out['value'] == task['correct'],
            'calls': calls, 'steps': steps,
            'latency': time.perf_counter() - t0}


def run_batch(config, tasks, seed):
    rng = random.Random(seed)
    return [solve_one(t, config, rng) for t in tasks]


def bench_config(config, tasks, seed):
    """在给定任务集上评估配置（不动全局配置）。"""
    rs = run_batch(config, tasks, seed)
    n = len(rs)
    return {
        'n_tasks': n,
        'success_rate': round(sum(r['success'] for r in rs) / n, 4),
        'avg_calls': round(sum(r['calls'] for r in rs) / n, 3),
        'avg_latency_ms': round(sum(r['latency'] for r in rs) / n * 1000, 3),
    }


# ---------- 引擎接线 ----------
AGENT_CONFIG = {'verify': False, 'retries': 0}   # 全局策略配置（进化对象）
CONFIG_HISTORY = [{'config': copy.deepcopy(AGENT_CONFIG), 'event': 'init'}]

memory = MemorySystem(store_path='/root/private_data/evo_memory.sqlite')
loop = EvolutionLoop(memory=memory, log_dir='/root/private_data/evo_logs')

CYCLE_RECORDS = []
SR_HISTORY = []


def observe_cycle(tag, tasks, seed, n=None):
    """跑一批任务 → 记录 trace → observe → evaluate，返回 (obs, assessment, sr)。"""
    sub = tasks[:n] if n else tasks
    t0 = time.time()
    results = run_batch(AGENT_CONFIG, sub, seed)
    wall = time.time() - t0
    sr = sum(r['success'] for r in results) / len(results)
    for task, r in zip(sub, results):
        loop.record_trace(TaskTrace(
            task_id=task['id'],
            description=f"{task['a']}{task['op']}{task['b']}={task['correct']} [{tag}]",
            start_time=t0, end_time=t0 + r['latency'],
            tools_used=['compute'],
            steps=r['steps'],
            success=r['success'],
            error='' if r['success'] else f"got {r['value']}, want {r['correct']}",
            metrics={'calls': r['calls']},
        ))
    obs = loop.observe()
    assessment = loop.evaluate(obs)
    SR_HISTORY.append({'cycle': tag, 'seed': seed, 'n': len(sub),
                       'success_rate': round(sr, 4), 'config': copy.deepcopy(AGENT_CONFIG)})
    CYCLE_RECORDS.append({'tag': tag, 'observation': obs,
                          'weaknesses': assessment['weaknesses'],
                          'overall_score': assessment['overall_score']})
    print(f"[{tag}] n={len(sub)} sr={sr:.3f} score={assessment['overall_score']:.3f} "
          f"weaknesses={[w['area'] for w in assessment['weaknesses']]}")
    return obs, assessment, sr


def apply_config(key, value, event):
    old = AGENT_CONFIG[key]
    AGENT_CONFIG[key] = value
    CONFIG_HISTORY.append({'config': copy.deepcopy(AGENT_CONFIG), 'event': event})
    return old


def make_improvement(layer, title, desc, rationale, gain, risk, changes,
                     bench_tasks, bench_seed, bench_threshold, baseline_sr):
    """构造一个真实的改进：沙箱/基准/独立评估全部实测。changes 为多键变更字典。"""
    def sandbox_fn():
        trial = copy.deepcopy(AGENT_CONFIG)
        trial.update(changes)
        b = bench_config(trial, bench_tasks[:10], bench_seed + 999)
        return {'status': 'ok', 'trial_sr_10': b['success_rate']}

    def benchmark_fn():
        trial = copy.deepcopy(AGENT_CONFIG)
        trial.update(changes)
        b = bench_config(trial, bench_tasks, bench_seed)
        calls_saved = round(BASELINE_CALLS - b['avg_calls'], 3)
        return {**b, 'calls_saved_vs_baseline': calls_saved,
                'baseline_sr': baseline_sr}

    def eval_fn():
        br = imp.benchmark_result
        passed = (br.get('success_rate', 0) >= bench_threshold
                  and 'error' not in br)
        return {'overall': bool(passed),
                'detail': f"bench_sr={br.get('success_rate')} threshold={bench_threshold}"}

    _old = {k: AGENT_CONFIG[k] for k in changes}

    def rollback_fn():
        for k, v in _old.items():
            AGENT_CONFIG[k] = v
        CONFIG_HISTORY.append({'config': copy.deepcopy(AGENT_CONFIG), 'event': f'rollback:{title}'})

    imp = loop.propose_improvement(
        layer=layer, title=title, description=desc,
        rationale=rationale, expected_gain=gain, risk_level=risk,
        rollback_fn=rollback_fn)
    loop.validate(imp, sandbox_fn=sandbox_fn, benchmark_fn=benchmark_fn, eval_fn=eval_fn)
    accepted = loop.decide(imp)
    if accepted:
        AGENT_CONFIG.update(changes)
        CONFIG_HISTORY.append({'config': copy.deepcopy(AGENT_CONFIG), 'event': f'accept:{title}'})
    return imp, accepted


# ---------- 点火 ----------
print('=== 阶段B首次点火：evolution_loop 完整闭环 ===')
BASELINE_CALLS = bench_config({'verify': False, 'retries': 0}, TASKS[:20], 11)['avg_calls']

# C0 基线
obs0, as0, sr0 = observe_cycle('C0_baseline', TASKS, seed=100, n=40)
assert sr0 < 0.55, f"基线成功率异常: {sr0}"

# ImpA：L3 策略层 —— 开启校验 + 重试2（0.45 → ~0.83）
impA, okA = make_improvement(
    3, 'Enable checksum verification with retries=2',
    'compute 工具 55% 概率返回损坏值，checksum 始终由真值计算；开启校验并重试可剔除损坏结果',
    'C0 基线 success_rate=0.45，弱点 severity=high',
    'success_rate 0.45 -> ~0.83', 'low',
    changes={'verify': True, 'retries': 2},
    bench_tasks=TASKS, bench_seed=700, bench_threshold=0.70, baseline_sr=sr0)
print(f"[ImpA] accepted={okA} bench={impA.benchmark_result.get('success_rate')} evidence={impA.evidence}")
assert okA, "ImpA 应被接受"

# C1
obs1, as1, sr1 = observe_cycle('C1_after_ImpA', TASKS, seed=200, n=40)

# ImpB：L3 策略层 —— 重试 2→4（0.83 → ~0.95）
impB, okB = make_improvement(
    3, 'Increase retries 2->4 for higher success rate',
    '残余失败来自连续损坏，增加独立重试次数进一步压低失败率',
    'C1 success_rate=0.83 仍低于 0.9 目标',
    'success_rate 0.83 -> ~0.95', 'low',
    changes={'retries': 4},
    bench_tasks=TASKS, bench_seed=700, bench_threshold=0.90, baseline_sr=sr1)
print(f"[ImpB] accepted={okB} bench={impB.benchmark_result.get('success_rate')} evidence={impB.evidence}")

# C2 → 提升为稳定版（ImpB 接受后 current_version 为 EXPERIMENTAL，promote 才会成功）
obs2, as2, sr2 = observe_cycle('C2_after_ImpB', TASKS, seed=300, n=40)
v_stable = loop.current_version.version_id
promoted = loop.promote_to_stable(v_stable)
assert promoted, "promote_to_stable 失败（当前版本非 EXPERIMENTAL）"
stable_sr = sr2
print(f"[promote] {v_stable} -> STABLE (sr={stable_sr:.3f})")

# ImpC：L6 工具层 —— 关闭校验省调用（应被证据拒绝）
impC, okC = make_improvement(
    6, 'Disable verification to save compute calls',
    '校验消耗额外调用；关闭可省算力',
    'C2 已达标，尝试降本',
    'avg_calls 下降', 'high',
    changes={'verify': False},
    bench_tasks=TASKS, bench_seed=700, bench_threshold=0.90, baseline_sr=stable_sr)
print(f"[ImpC] accepted={okC} bench={impC.benchmark_result.get('success_rate')} evidence={impC.evidence} (期望 rejected)")

# ImpD：L4 流程层 —— 重试 4→2 省算力。基准子集阈值有缺陷（只查 30 任务子集 sr>=0.75）
#   子集上 retry2 的 sr≈0.77 能过松阈值 → 被放行 → 但全量新种子会暴露回归
impD, okD = make_improvement(
    4, 'Reduce retries 4->2 to save compute',
    '每次任务平均调用数偏高，削减重试换算力',
    'C2 avg_calls 偏高，尝试降本',
    'avg_calls 5.0 -> 3.0', 'medium',
    changes={'retries': 2},
    bench_tasks=TASKS[:30], bench_seed=800, bench_threshold=0.75, baseline_sr=stable_sr)
print(f"[ImpD] accepted={okD} bench={impD.benchmark_result.get('success_rate')} evidence={impD.evidence} (基准有缺陷，被放行)")
assert okD, "ImpD 应被有缺陷的基准放行（以演示回滚）"

# C3：全量、新种子 → 检出回归
obs3, as3, sr3 = observe_cycle('C3_after_ImpD_fullsuite', TASKS, seed=400)
regression = stable_sr - sr3
rolled_back = False
if regression > 0.05:
    print(f"[regression] sr {stable_sr:.3f} -> {sr3:.3f} (drop {regression:.3f}) → 触发回滚")
    rolled_back = loop.rollback()
    memory.store_semantic(
        f"教训：基准子集阈值放行了坏改进[{impD.title}]，全量回归 {regression:.2f}。"
        f"改进项评估必须包含全量新种子回归检查（drop>5pp 即回滚）。",
        source='evolution_loop', importance=0.95,
        tags=['evolution', 'lesson', 'benchmark_criteria'], evidence='VERIFIED')
print(f"[rollback] done={rolled_back} config={AGENT_CONFIG}")

# C4：恢复确认
obs4, as4, sr4 = observe_cycle('C4_after_rollback', TASKS, seed=500)
recovered = sr4 >= stable_sr - 0.05

# 健康态跑一次引擎自主 run_cycle（应报告无候选改进）
idle_cycle = loop.run_cycle()

# 记忆固化
memory.store_semantic(
    f"阶段B首跑结论：{len(SR_HISTORY)}个观测周期，"
    f"sr {sr0:.2f}->{sr4:.2f}；接受{sum(1 for i in loop.improvements if i.status=='accepted')}项/"
    f"拒绝{sum(1 for i in loop.improvements if i.status=='rejected')}项；"
    f"回滚{1 if rolled_back else 0}次；稳定版={loop.stable_version.version_id if loop.stable_version else None}",
    source='evolution_loop', importance=0.9, tags=['evolution', 'phase_b', 'first_fire'], evidence='VERIFIED')
try:
    consol = memory.consolidate()
except Exception as e:
    consol = {'error': str(e)}

# ---------- 汇总 ----------
stats = loop.stats()
result = {
    'ok': True,
    'sr_history': SR_HISTORY,
    'config_history': CONFIG_HISTORY,
    'improvements': [{
        'id': i.id, 'layer': f'L{i.layer}', 'title': i.title,
        'status': i.status, 'evidence': i.evidence,
        'sandbox': i.sandbox_result, 'benchmark': i.benchmark_result,
        'eval': i.eval_result,
        'version_before': i.version_before, 'version_after': i.version_after,
    } for i in loop.improvements],
    'versions': [{
        'version_id': v.version_id, 'status': v.status,
        'description': v.description, 'parent': v.parent_version,
    } for v in loop.versions],
    'rollback': {'triggered': rolled_back, 'regression': round(regression, 4)},
    'recovered': bool(recovered),
    'stable_version': loop.stable_version.version_id if loop.stable_version else None,
    'stats': stats,
    'memory_consolidate': consol if isinstance(consol, dict) else {'result': str(consol)},
    'idle_cycle_actions': idle_cycle.get('actions', 'no_candidates'),
    'audit_trail_tail': loop.get_audit_trail(12),
    'wall_s': None,
}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1, default=str)

print('\n=== 点火结果 ===')
sr_line = ' -> '.join('{:.2f}'.format(h['success_rate']) for h in SR_HISTORY)
print(f"sr 轨迹: {sr_line}")
print(f"改进: 接受 {[i.title[:30] for i in loop.improvements if i.status == 'accepted']}")
print(f"      拒绝 {[i.title[:30] for i in loop.improvements if i.status == 'rejected']}")
print(f"回滚: {rolled_back} | 恢复: {recovered} | 稳定版: {result['stable_version']}")
print(f"stats: {json.dumps(stats, ensure_ascii=False)}")
print(f"结果已写入 {OUT}")
