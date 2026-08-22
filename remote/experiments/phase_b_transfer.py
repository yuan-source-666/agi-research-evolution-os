# -*- coding: utf-8 -*-
"""阶段 B 第二步：跨任务族迁移实验（NFL 判定点）+ 回归门禁内化验证。

任务族 A（静默损坏）：compute 55% 概率返回损坏值，checksum 恒由真值计算
  → verify（校验）可检测损坏，retries 配合 verify 有效。
任务族 B（显性故障）：compute 55% 概率直接报错（无返回值）
  → 错误天然可感知，retries 单独有效；verify 检测不到任何东西，纯开销。

流程：
  Loop-A 在 A 上进化 -> A*（verify+retries）；
  Loop-B 在 B 上从零进化 -> B*（retries only，含门禁拦截演示）；
  迁移评估：{base, A*, B*} x {A, B} 全矩阵（600 任务全新种子）。
"""
import copy
import json
import random
import sys
import time

sys.path.insert(0, '/root/private_data')

from memory_system import MemorySystem
from evolution_loop import EvolutionLoop, TaskTrace

P_FAIL = 0.55
N_TASKS = 120
EVAL_N = 600


def make_tasks(n, seed=7):
    r = random.Random(seed)
    ops = ['+', '-', '*']
    tasks = []
    for i in range(n):
        a, b = r.randint(10, 99), r.randint(2, 9)
        op = r.choice(ops)
        correct = a + b if op == '+' else (a - b if op == '-' else a * b)
        tasks.append({'id': 't%03d' % i, 'a': a, 'b': b, 'op': op, 'correct': correct})
    return tasks


def checksum(task, value):
    return (task['a'] * 1000003 + task['b'] * 91 + ord(task['op']) * 7 + value * 13) & 0xffff


def tool_call(task, rng, family):
    """family='A' 静默损坏（checksum 恒真值）；family='B' 显性报错。"""
    correct = task['correct']
    if family == 'A':
        corrupted = rng.random() < P_FAIL
        value = correct + (rng.randint(1, 9) * 10 if corrupted else 0)
        return {'value': value, 'checksum': checksum(task, correct)}
    if rng.random() < P_FAIL:
        return {'error': True}
    return {'value': correct, 'checksum': checksum(task, correct)}


def solve_one(task, config, rng, family):
    t0 = time.perf_counter()
    steps, calls, out = [], 0, None
    for attempt in range(1 + config['retries']):
        r = tool_call(task, rng, family)
        calls += 1
        if r.get('error'):
            steps.append({'action': 'compute#%d' % attempt, 'result': 'ERROR',
                          'success': False})
            continue
        out = r
        if config['verify']:
            calls += 1  # 校验本身是一次独立的工具调用
            ok = checksum(task, r['value']) == r['checksum']
        else:
            ok = True
        steps.append({'action': 'compute#%d' % attempt, 'result': r['value'],
                      'success': ok})
        if ok:
            break
    success = out is not None and out['value'] == task['correct']
    return {'value': out['value'] if out else None, 'correct': task['correct'],
            'success': success, 'calls': calls, 'steps': steps,
            'latency': time.perf_counter() - t0}


def run_batch(config, tasks, seed, family):
    rng = random.Random(seed)
    return [solve_one(t, config, rng, family) for t in tasks]


def bench_config(config, tasks, seed, family):
    rs = run_batch(config, tasks, seed, family)
    n = len(rs)
    return {'n_tasks': n,
            'success_rate': round(sum(r['success'] for r in rs) / n, 4),
            'avg_calls': round(sum(r['calls'] for r in rs) / n, 3)}


# ============ 单族进化 ============

def evolve_family(family, tasks, log_dir, mem_path):
    cfg = {'verify': False, 'retries': 0}
    memory = MemorySystem(store_path=mem_path)
    loop = EvolutionLoop(memory=memory, log_dir=log_dir)

    gate = {'trial': None, 'base_cfg': None, 'seed': 4242, 'thr': 0.80}

    def gate_fn(imp):
        trial = gate['trial']
        if trial is None:
            return {'overall': True, 'detail': 'no trial registered'}
        b = bench_config(trial, tasks, gate['seed'], family)
        base = bench_config(gate['base_cfg'], tasks, gate['seed'] + 1, family)
        ok = (b['success_rate'] >= gate['thr']
              and b['success_rate'] >= base['success_rate'] - 0.02)
        return {'overall': bool(ok),
                'detail': 'full-suite trial_sr=%s base_sr=%s thr=%s avg_calls=%s'
                          % (b['success_rate'], base['success_rate'],
                             gate['thr'], b['avg_calls']),
                'trial_bench': b}

    loop.set_regression_gate(gate_fn)

    sr_history = []

    def observe_cycle(tag, seed, n=None):
        sub = tasks[:n] if n else tasks
        t0 = time.time()
        results = run_batch(cfg, sub, seed, family)
        sr = sum(r['success'] for r in results) / len(results)
        for task, r in zip(sub, results):
            loop.record_trace(TaskTrace(
                task_id='%s_%s' % (family, task['id']),
                description='%d%s%d=%d [%s]' % (task['a'], task['op'], task['b'],
                                                task['correct'], tag),
                start_time=t0, end_time=t0 + r['latency'],
                tools_used=['compute'], steps=r['steps'], success=r['success'],
                error='' if r['success'] else 'got %s want %s' % (r['value'], r['correct']),
                metrics={'calls': r['calls']}))
        obs = loop.observe()
        assessment = loop.evaluate(obs)
        sr_history.append({'cycle': tag, 'n': len(sub), 'success_rate': round(sr, 4),
                           'config': copy.deepcopy(cfg)})
        print('[%s/%s] n=%d sr=%.3f score=%.3f' %
              (family, tag, len(sub), sr, assessment['overall_score']))
        return obs, assessment, sr

    def make_improvement(layer, title, desc, rationale, changes,
                         bench_tasks, bench_seed, bench_threshold,
                         min_gain, baseline_sr, flawed=False):
        """真实改进流程：沙箱/基准实测 -> validate -> 回归门禁 -> decide。"""
        trial = copy.deepcopy(cfg)
        trial.update(changes)

        def sandbox_fn():
            b = bench_config(trial, tasks[:10], bench_seed + 999, family)
            return {'status': 'ok', 'trial_sr_10': b['success_rate']}

        def benchmark_fn():
            b = bench_config(trial, bench_tasks, bench_seed, family)
            return {'success_rate': b['success_rate'], 'avg_calls': b['avg_calls'],
                    'n_tasks': b['n_tasks'],
                    'baseline_sr': baseline_sr,
                    'note': 'flawed subset benchmark' if flawed else 'full benchmark'}

        def eval_fn():
            br = imp.benchmark_result
            gain = br.get('success_rate', 0) - baseline_sr
            passed = (br.get('success_rate', 0) >= bench_threshold
                      and 'error' not in br and gain >= min_gain)
            return {'overall': bool(passed),
                    'detail': 'bench_sr=%s baseline=%s gain=%s need_thr=%s need_gain=%s'
                              % (br.get('success_rate'), baseline_sr, round(gain, 4),
                                 bench_threshold, min_gain)}

        def rollback_fn():
            for k, v in changes.items():
                cfg[k] = saved[k]

        saved = copy.deepcopy(cfg)
        gate['trial'] = copy.deepcopy(trial)
        gate['base_cfg'] = copy.deepcopy(cfg)

        imp = loop.propose_improvement(
            layer=layer, title=title, description=desc, rationale=rationale,
            expected_gain='see benchmark', risk_level='medium',
            rollback_fn=rollback_fn)
        loop.validate(imp, sandbox_fn=sandbox_fn, benchmark_fn=benchmark_fn,
                      eval_fn=eval_fn)
        accepted = loop.decide(imp)
        if accepted:
            cfg.update(changes)
        return imp, accepted

    # --- 基线 ---
    observe_cycle('C0_baseline', seed=100)

    records = []
    proposals = PROPOSALS[family]
    for i, p in enumerate(proposals):
        base_sr = sr_history[-1]['success_rate']
        imp, ok = make_improvement(
            p['layer'], p['title'], p['desc'], p['rationale'],
            p['changes'], p['bench_tasks'](tasks), p['bench_seed'],
            p['bench_threshold'], p['min_gain'], base_sr, p.get('flawed', False))
        gate_info = (imp.eval_result or {}).get('regression_gate')
        records.append({'title': p['title'], 'accepted': ok,
                        'evidence': imp.evidence,
                        'bench_sr': (imp.benchmark_result or {}).get('success_rate'),
                        'gate': gate_info})
        print('[%s/Imp%d] accepted=%s evidence=%s bench=%s gate=%s' %
              (family, i + 1, ok, imp.evidence,
               (imp.benchmark_result or {}).get('success_rate'),
               (gate_info or {}).get('detail', '-')))
        observe_cycle('C%d_after_Imp%d' % (i + 1, i + 1), seed=200 + i)

    # 提升稳定版
    if loop.current_version:
        loop.promote_to_stable(loop.current_version.version_id)

    return {'family': family, 'final_config': copy.deepcopy(cfg),
            'sr_history': sr_history, 'improvements': records,
            'n_accepted': sum(1 for r in records if r['accepted']),
            'n_rejected': sum(1 for r in records if not r['accepted']),
            'gate_blocks': [r for r in records
                            if r['gate'] and not r['gate'].get('overall')]}


PROPOSALS = {
    'A': [
        {'layer': 3, 'title': 'Enable checksum verification with retries=2',
         'desc': '静默损坏环境下开启校验并重试',
         'rationale': 'C0 基线 sr=0.45，损坏无法自检',
         'changes': {'verify': True, 'retries': 2},
         'bench_tasks': lambda t: t, 'bench_seed': 700,
         'bench_threshold': 0.70, 'min_gain': 0.10},
        {'layer': 3, 'title': 'Raise retries 2->4',
         'desc': '残余损坏再降一档',
         'rationale': 'C1 后仍有 0.55^3 残余失败率',
         'changes': {'retries': 4},
         'bench_tasks': lambda t: t, 'bench_seed': 710,
         'bench_threshold': 0.85, 'min_gain': 0.02},
        {'layer': 4, 'title': 'Disable verification to save calls',
         'desc': '省掉每次校验的额外调用',
         'rationale': 'avg_calls 偏高，尝试降本',
         'changes': {'verify': False},
         'bench_tasks': lambda t: t, 'bench_seed': 720,
         'bench_threshold': 0.85, 'min_gain': 0.02},
    ],
    'B': [
        {'layer': 3, 'title': 'Raise retries 0->4 (errors are loud)',
         'desc': '显性报错可直接重试',
         'rationale': 'C0 基线 sr=0.45，错误天然可感知',
         'changes': {'retries': 4},
         'bench_tasks': lambda t: t, 'bench_seed': 700,
         'bench_threshold': 0.70, 'min_gain': 0.10},
        {'layer': 3, 'title': 'Enable verification for safety',
         'desc': '给成功结果加校验（B 族无静默损坏，预期无增益）',
         'rationale': '来自 A 族经验的策略迁移尝试',
         'changes': {'verify': True},
         'bench_tasks': lambda t: t, 'bench_seed': 710,
         'bench_threshold': 0.70, 'min_gain': 0.05},
        {'layer': 4, 'title': 'Reduce retries 4->0 to save compute',
         'desc': '削减重试换算力（基准子集阈值过松的缺陷设计）',
         'rationale': 'avg_calls 偏高',
         'changes': {'retries': 0},
         'bench_tasks': lambda t: t[:20], 'bench_seed': 730,
         'bench_threshold': 0.40, 'min_gain': -1.0, 'flawed': True},
    ],
}


def main():
    tasks = make_tasks(N_TASKS)

    print('==== Loop-A: evolve on family A (silent corruption) ====')
    resA = evolve_family('A', tasks, '/root/private_data/evo_logs_A',
                         '/root/private_data/evo_memory_A.sqlite')
    print('A* =', resA['final_config'])

    print('==== Loop-B: evolve on family B (loud errors) ====')
    resB = evolve_family('B', tasks, '/root/private_data/evo_logs_B',
                         '/root/private_data/evo_memory_B.sqlite')
    print('B* =', resB['final_config'])

    # ---- 迁移矩阵 ----
    evalA = make_tasks(EVAL_N, seed=5001)
    evalB = make_tasks(EVAL_N, seed=5002)
    configs = {
        'base': {'verify': False, 'retries': 0},
        'A_star': resA['final_config'],
        'B_star': resB['final_config'],
    }
    matrix = []
    for cname, c in configs.items():
        for fam, ev in (('A', evalA), ('B', evalB)):
            b = bench_config(c, ev, 9001 if fam == 'A' else 9002, fam)
            matrix.append({'config': cname, 'config_dict': copy.deepcopy(c),
                           'family': fam, 'n': EVAL_N,
                           'success_rate': b['success_rate'],
                           'avg_calls': b['avg_calls']})
            print('[matrix] %-7s on %s: sr=%.4f avg_calls=%.3f' %
                  (cname, fam, b['success_rate'], b['avg_calls']))

    # 迁移指标
    def m(c, f):
        for row in matrix:
            if row['config'] == c and row['family'] == f:
                return row
        raise KeyError((c, f))

    transfer = {
        'A_star_on_B': {
            'sr': m('A_star', 'B')['success_rate'],
            'native_sr': m('B_star', 'B')['success_rate'],
            'sr_ratio': round(m('A_star', 'B')['success_rate'] /
                              max(m('B_star', 'B')['success_rate'], 1e-9), 4),
            'call_overhead': round(m('A_star', 'B')['avg_calls'] /
                                   max(m('B_star', 'B')['avg_calls'], 1e-9), 3)},
        'B_star_on_A': {
            'sr': m('B_star', 'A')['success_rate'],
            'native_sr': m('A_star', 'A')['success_rate'],
            'sr_ratio': round(m('B_star', 'A')['success_rate'] /
                              max(m('A_star', 'A')['success_rate'], 1e-9), 4)},
    }

    result = {
        'ok': True,
        'p_fail': P_FAIL,
        'n_tasks_train': N_TASKS,
        'n_tasks_eval': EVAL_N,
        'familyA': resA,
        'familyB': resB,
        'matrix': matrix,
        'transfer': transfer,
        'gate_blocked_titles': [r['title'] for r in resB['gate_blocks']],
        'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    out = '/root/private_data/phase_b_transfer.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print('==== saved %s ====' % out)
    print('A*=%s B*=%s' % (resA['final_config'], resB['final_config']))
    print('transfer A*->B:', transfer['A_star_on_B'])
    print('transfer B*->A:', transfer['B_star_on_A'])
    print('gate blocked:', result['gate_blocked_titles'])


if __name__ == '__main__':
    main()
