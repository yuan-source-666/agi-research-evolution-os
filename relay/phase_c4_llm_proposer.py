# -*- coding: utf-8 -*-
"""阶段 C v4：候选种群 + 门禁选择（生成/选择分离）。

v3 病根：候选集里明明有能过门禁的 {verify:False, retries:3}，
但"取首个合法"策略抓了最弱的 retries=1，5 轮全军覆没。
v4 修复：每轮采样 k=6 个候选 -> 去重 -> 每个候选先用独立种子做全量
预筛（trial sr >= 0.80 且 >= base-0.02）-> 选预筛通过且 trial sr 最高者
正式走 make_improvement 链（沙箱/基准/门禁/decide 证据链不变）。
预筛淘汰的候选连同实测数据记入 HISTORY 反馈。
"""
import copy
import json
import random
import re
import sys
import time

sys.path.insert(0, '/root/private_data')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_system import MemorySystem
from evolution_loop import EvolutionLoop, TaskTrace

MODEL_PATH = "/root/private_data/Qwen2.5-1.5B-Instruct"
P_FAIL = 0.55
N_TASKS = 120
EVAL_N = 600
N_ROUNDS = 5
K_SAMPLES = 6
OUT_JSON = '/root/private_data/phase_c4_llm_proposer.json'

RESULT = {'phase': 'C4_llm_proposer_pop_select',
          'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
          'model': MODEL_PATH, 'p_fail': P_FAIL, 'n_tasks': N_TASKS,
          'n_rounds': N_ROUNDS, 'k_samples': K_SAMPLES, 'families': {}}


def save():
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(RESULT, f, ensure_ascii=False, indent=1)


# ============ 任务族（同前） ============

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
            calls += 1
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


# ============ LLM 提案器（best-of-k） ============

class Proposer:
    def __init__(self):
        print('[proposer] loading tokenizer...', flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        print('[proposer] loading model (bf16, cuda, eager)...', flush=True)
        t0 = time.time()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, dtype=torch.bfloat16, device_map='cuda',
                attn_implementation='eager')
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16, device_map='cuda',
                attn_implementation='eager')
        self.model.eval()
        free, total = torch.cuda.mem_get_info(0)
        print('[proposer] loaded %.1fs, vram=%.1fGB free=%.1fGB' %
              (time.time() - t0, torch.cuda.memory_allocated(0) / 1e9, free / 1e9),
              flush=True)

    def chat(self, user, temperature=0.8, max_new_tokens=220):
        text = self.tok.apply_chat_template(
            [{'role': 'user', 'content': user}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=max(temperature, 0.05), top_p=0.95,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.decode(out[0][inputs['input_ids'].shape[1]:],
                               skip_special_tokens=True)


def build_prompt(family, cfg, stats, fail_samples, ok_sample, history):
    lines = [
        'You are the PROPOSAL module of a self-evolving agent.',
        'Diagnose the failure pattern from the traces, then output ONE improvement proposal.',
        '',
        'ENVIRONMENT',
        '- Tool "compute" answers arithmetic tasks. '
        'Family B: the tool sometimes crashes with a loud error.',
        '- Agent config knobs:',
        '  verify (true/false): if true, after each compute call the agent checks the',
        '  returned value against an independent checksum; on mismatch it discards the',
        '  result and retries. If the tool only fails with LOUD errors, verification adds',
        '  overhead but cannot catch anything.',
        '  retries (0-8): number of extra compute calls allowed after a failed or invalid attempt.',
        '- Each call fails independently with probability about 0.55. With retries=N, the',
        '  task fails only if all N+1 calls fail.',
        '',
        'HOW TO READ TRACES',
        '- A step with "result": "ERROR" means the tool crashed loudly (retrying works).',
        '- All steps "success": true but the final answer is WRONG means SILENT CORRUPTION:',
        '  the tool returned a wrong value that looks valid (only verification catches it).',
        '',
        'CURRENT CONFIG: ' + json.dumps(cfg),
        '',
        'LAST CYCLE STATISTICS',
        '- success_rate: %.3f' % stats['success_rate'],
        '- avg tool calls per task: %.2f' % stats['avg_calls'],
        '',
        'FAILED TASK EXAMPLES',
    ]
    for i, (task, r) in enumerate(fail_samples):
        steps = json.dumps(r['steps'])
        lines.append('%d. task "%d%s%d=%d": steps=%s final=WRONG (got %s, want %d)'
                     % (i + 1, task['a'], task['op'], task['b'], task['correct'],
                        steps, r['value'], task['correct']))
    if ok_sample is not None:
        task, r = ok_sample
        lines.append('')
        lines.append('SUCCEEDED EXAMPLE')
        lines.append('task "%d%s%d=%d": steps=%s final=OK'
                     % (task['a'], task['op'], task['b'], task['correct'],
                        json.dumps(r['steps'])))
    if history:
        lines += [
            '',
            'PROPOSAL HISTORY (this session)',
            '- Do NOT repeat an already-applied or rejected proposal.',
            '- If a proposal was REJECTED, the reason tells you what to fix: propose a',
            '  STRONGER or DIFFERENT change.',
        ]
        for h in history:
            lines.append('- round %d: proposal %s -> %s (%s)' %
                         (h['round'], json.dumps(h['changes']),
                          'ACCEPTED' if h['accepted'] else 'REJECTED', h['reason']))
    lines += [
        '',
        'OUTPUT',
        'Output ONLY one JSON object, no other text:',
        '{"title": "short title", "rationale": "one sentence diagnosis", '
        '"changes": {"verify": true or false, "retries": integer 0-8}}',
    ]
    return '\n'.join(lines)


def parse_proposal(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None, 'no JSON object found'
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        return None, 'json parse error: %s' % e
    if not isinstance(obj, dict) or 'changes' not in obj:
        return None, 'missing "changes" key'
    ch = obj['changes']
    if not isinstance(ch, dict):
        return None, '"changes" is not an object'
    out = {}
    if 'verify' in ch:
        v = ch['verify']
        if isinstance(v, str):
            v = v.strip().lower() in ('true', '1', 'yes')
        if not isinstance(v, bool):
            return None, 'verify is not bool'
        out['verify'] = v
    if 'retries' in ch:
        try:
            rt = int(ch['retries'])
        except Exception:
            return None, 'retries is not int'
        if not 0 <= rt <= 8:
            return None, 'retries out of range 0-8'
        out['retries'] = rt
    if not out:
        return None, 'changes has no valid keys'
    obj['changes'] = out
    obj.setdefault('title', 'untitled')
    obj.setdefault('rationale', '')
    return obj, None


def is_noop(changes, cfg):
    return all(cfg.get(k) == v for k, v in changes.items())


def ask_proposal_bestofk(proposer, family, cfg, stats, fail_samples, ok_sample,
                         history):
    """每轮采样 K_SAMPLES 个候选，过滤 no-op/重复，取首个合法提案。"""
    prompt = build_prompt(family, cfg, stats, fail_samples, ok_sample, history)
    rejected_before = [h['changes'] for h in history if not h['accepted']]
    accepted_before = [h['changes'] for h in history if h['accepted']]
    samples = []
    chosen = None
    for k in range(K_SAMPLES):
        t0 = time.time()
        raw = proposer.chat(prompt, temperature=0.8)
        lat = round(time.time() - t0, 2)
        prop, err = parse_proposal(raw)
        note = err
        if prop is not None:
            if is_noop(prop['changes'], cfg):
                note = 'no-op'
            elif prop['changes'] in rejected_before:
                note = 'duplicate of rejected'
            elif prop['changes'] in accepted_before:
                note = 'duplicate of accepted'
            else:
                note = 'VALID'
                if chosen is None:
                    chosen = prop
        samples.append({'k': k + 1, 'latency_s': lat, 'changes':
                        (prop['changes'] if prop else None), 'note': note})
        print('[llm/%s/s%d] %.1fs -> %s [%s]' %
              (family, k + 1, lat,
               (prop['changes'] if prop else err), note), flush=True)
    return chosen, samples


# ============ B 族进化（best-of-k） ============

def evolve_family_llm(family, tasks, proposer):
    cfg = {'verify': False, 'retries': 0}
    memory = MemorySystem(store_path='/root/private_data/evo_memory_C4_%s.sqlite' % family)
    loop = EvolutionLoop(memory=memory,
                         log_dir='/root/private_data/evo_logs_C4_%s' % family)

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
    last_results = [None]
    history = []

    def observe_cycle(tag, seed, n=None):
        sub = tasks[:n] if n else tasks
        t0 = time.time()
        results = run_batch(cfg, sub, seed, family)
        sr = sum(r['success'] for r in results) / len(results)
        for task, r in zip(sub, results):
            loop.record_trace(TaskTrace(
                task_id='%sC4_%s' % (family, task['id']),
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
        last_results[0] = results
        print('[%s/%s] sr=%.3f score=%.3f' % (family, tag, sr, assessment['overall_score']),
              flush=True)
        return sr

    def make_improvement(layer, title, desc, rationale, changes,
                         bench_seed, bench_threshold, min_gain, baseline_sr):
        trial = copy.deepcopy(cfg)
        trial.update(changes)

        def sandbox_fn():
            b = bench_config(trial, tasks[:10], bench_seed + 999, family)
            return {'status': 'ok', 'trial_sr_10': b['success_rate']}

        def benchmark_fn():
            b = bench_config(trial, tasks, bench_seed, family)
            return {'success_rate': b['success_rate'], 'avg_calls': b['avg_calls'],
                    'n_tasks': b['n_tasks'], 'baseline_sr': baseline_sr}

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

    fam_state = {'family': family, 'sr_history': [], 'proposals': [],
                 'final_config': None}

    observe_cycle('C0_baseline', seed=100)

    for rnd in range(1, N_ROUNDS + 1):
        base_sr = sr_history[-1]['success_rate']
        stats = {'success_rate': base_sr,
                 'avg_calls': round(sum(r['calls'] for r in last_results[0]) /
                                    len(last_results[0]), 2)}
        results = last_results[0]
        fails = [(t, r) for t, r in zip(tasks, results) if not r['success']][:4]
        oks = [(t, r) for t, r in zip(tasks, results) if r['success']][:1]
        ok_sample = oks[0] if oks else None

        prop, samples = ask_proposal_bestofk(
            proposer, family, copy.deepcopy(cfg), stats, fails, ok_sample, history)
        rec = {'round': rnd, 'base_sr': base_sr, 'samples': samples,
               'proposal': prop, 'accepted': None,
               'bench_sr': None, 'gate': None, 'parse_ok': prop is not None,
               'prescreen': []}

        # ---- 种群预筛：每个合法候选独立种子全量实测 ----
        seen = set()
        candidates = []
        for s in samples:
            ch = s.get('changes')
            if ch is None or s.get('note') != 'VALID':
                continue
            key = json.dumps(ch, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            trial = copy.deepcopy(cfg)
            trial.update(ch)
            # 预筛种子独立于正式门禁种子(4242/4243)，避免选择偏置
            pb = bench_config(trial, tasks, 8000 + rnd * 13, family)
            ok_pre = (pb['success_rate'] >= 0.80
                      and pb['success_rate'] >= base_sr - 0.02)
            rec['prescreen'].append({'changes': ch,
                                     'prescreen_sr': pb['success_rate'],
                                     'avg_calls': pb['avg_calls'],
                                     'prescreen_pass': bool(ok_pre)})
            print('[pre/%s/R%d] %s trial_sr=%s pass=%s' %
                  (family, rnd, ch, pb['success_rate'], ok_pre), flush=True)
            if ok_pre:
                candidates.append((pb['success_rate'], ch))
            else:
                history.append({'round': rnd, 'changes': ch, 'accepted': False,
                                'reason': 'prescreen sr=%s below 0.80' %
                                          pb['success_rate']})
        candidates.sort(key=lambda x: -x[0])

        if not candidates:
            print('[%s/R%d] no prescreen-passing candidate, round skipped' %
                  (family, rnd), flush=True)
            if prop is not None:
                rec['accepted'] = False
                rec['prescreen_all_failed'] = True
            fam_state['proposals'].append(rec)
            if prop is None:
                history.append({'round': rnd, 'changes': None, 'accepted': False,
                                'reason': 'no valid candidate'})
            RESULT['families'][family] = fam_state
            save()
            continue

        # 选预筛通过且 trial_sr 最高的候选正式入链（生成/选择分离）
        best_sr, best_changes = candidates[0]
        prop['changes'] = best_changes
        rec['selected_by'] = 'prescreen_best sr=%s' % best_sr

        bench_threshold = round(min(base_sr + 0.05, 0.97), 2)
        imp, ok = make_improvement(
            layer=3, title=prop['title'],
            desc=json.dumps(prop['changes']),
            rationale=prop['rationale'],
            changes=prop['changes'],
            bench_seed=700 + rnd * 10,
            bench_threshold=bench_threshold, min_gain=0.01,
            baseline_sr=base_sr)
        gate_info = (imp.eval_result or {}).get('regression_gate')
        rec['accepted'] = ok
        rec['bench_sr'] = (imp.benchmark_result or {}).get('success_rate')
        rec['gate'] = gate_info
        rec['evidence'] = imp.evidence
        if ok:
            reason = 'accepted: bench_sr=%s' % rec['bench_sr']
        elif gate_info and not gate_info.get('overall'):
            reason = 'REJECTED by regression gate: %s' % gate_info.get('detail')
        else:
            reason = 'REJECTED by eval: %s' % ((imp.eval_result or {}).get('detail', '-'))
        history.append({'round': rnd, 'changes': prop['changes'],
                        'accepted': ok, 'reason': reason})
        print('[%s/R%d] %s accepted=%s bench=%s' %
              (family, rnd, prop['changes'], ok, rec['bench_sr']), flush=True)
        fam_state['proposals'].append(rec)

        observe_cycle('C%d_after_R%d' % (rnd, rnd), seed=200 + rnd)
        RESULT['families'][family] = fam_state
        save()

    if loop.current_version:
        loop.promote_to_stable(loop.current_version.version_id)

    fam_state['final_config'] = copy.deepcopy(cfg)
    fam_state['sr_history'] = sr_history
    fam_state['n_accepted'] = sum(1 for r in fam_state['proposals'] if r['accepted'])
    fam_state['n_parse_ok'] = sum(1 for r in fam_state['proposals'] if r['parse_ok'])
    RESULT['families'][family] = fam_state
    save()
    return fam_state


def main():
    print('==== Phase C v4: population + gate selection (family B) ====', flush=True)
    proposer = Proposer()

    tasks = make_tasks(N_TASKS)
    resB = evolve_family_llm('B', tasks, proposer)
    print('B_llm4* =', resB['final_config'], flush=True)

    evalA = make_tasks(EVAL_N, seed=6001)
    evalB = make_tasks(EVAL_N, seed=6002)
    configs = {
        'base': {'verify': False, 'retries': 0},
        'B_llm4_star': resB['final_config'],
        'B_manual_star': {'verify': False, 'retries': 4},
    }
    matrix = []
    for cname, c in configs.items():
        for fam, ev in (('A', evalA), ('B', evalB)):
            b = bench_config(c, ev, 9101 if fam == 'A' else 9102, fam)
            matrix.append({'config': cname, 'config_dict': copy.deepcopy(c),
                           'family': fam, 'n': EVAL_N,
                           'success_rate': b['success_rate'],
                           'avg_calls': b['avg_calls']})
            print('[matrix] %-14s on %s: sr=%.4f avg_calls=%.3f' %
                  (cname, fam, b['success_rate'], b['avg_calls']), flush=True)
    RESULT['matrix'] = matrix
    RESULT['ok'] = True
    save()
    print('==== saved %s ====' % OUT_JSON, flush=True)
    print('PHASE_C4_DONE', flush=True)


if __name__ == '__main__':
    main()
