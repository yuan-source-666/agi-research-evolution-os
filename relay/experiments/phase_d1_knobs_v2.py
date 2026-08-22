# -*- coding: utf-8 -*-
"""G1 / D1: 10-knob sandbox over 5 task families.

进化对象从 2 旋钮 {verify, retries} 升到 10 旋钮。
三基线对照: default / random-search(200) / 2-knob-exhaustive(18)。
LLM 提案器: Qwen2.5-1.5B-Instruct, 种群生成(k=6) + 门禁预筛选择 (Phase C v4 机制)。

G1 验收 (代价感知版):
  score = sr - cost(cfg), cost 由旋钮组合决定 (重旋钮扣分):
  0) evo final score > random-search best score
  1) evo final score > 2-knob exhaustive best score
  2) 被选中旋钮与因果杠杆表重合 (报告人工对照)
"""
import json, random, copy, os, sys, time, re

FAMILIES = ['A', 'B', 'C', 'D', 'E']
N_TASKS = 120
BASE_SEED = 20260822
GATE_SEED = 777001
FINAL_SEED = 424242
K_SAMPLES = 6
N_ROUNDS = 5
MIN_GAIN = 0.02
MODEL_PATH = '/root/private_data/Qwen2.5-1.5B-Instruct'

# ---- 代价模型: 旋钮不是免费的, 重配置扣分 ----
TOOL_COST = {'fast_risky': 0.0, 'slow_safe': 0.04, 'both': 0.07}
COST_FORMULA = ('cost = 0.04*verify_depth + 0.012*retries + 0.03*task_decompose '
                '+ 0.04*multi_path_voting + 0.02*memory_read + 0.02*memory_write '
                '+ 0.015*max(0, timeout_budget-2) - 0.01*early_exit '
                '+ 0.005*diversity_seed + (fast_risky=0, slow_safe=0.04, both=0.07)')


def cfg_cost(cfg):
    c = (0.04 * cfg['verify_depth'] + 0.012 * cfg['retries']
         + 0.03 * cfg['task_decompose'] + 0.04 * cfg['multi_path_voting']
         + 0.02 * cfg['memory_read'] + 0.02 * cfg['memory_write']
         + 0.015 * max(0, cfg['timeout_budget'] - 2)
         + (-0.01) * cfg['early_exit']
         + 0.005 * cfg['diversity_seed']
         + TOOL_COST.get(cfg['tool_choice'], 0.0))
    return round(max(0.0, c), 4)


def score_of(sr, cfg):
    return sr - cfg_cost(cfg)

KNOB_SPEC = {
    'verify_depth':      [0, 1, 2],
    'retries':           [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'task_decompose':    [0, 1, 2],
    'multi_path_voting': [0, 3, 5],
    'memory_read':       [0, 1],
    'memory_write':      [0, 1],
    'timeout_budget':    [1, 2, 3, 4, 6, 8],
    'early_exit':        [0, 1],
    'tool_choice':       ['fast_risky', 'slow_safe', 'both'],
    'diversity_seed':    [0, 1, 2, 3],
}

DEFAULT_CFG = {
    'verify_depth': 0, 'retries': 0, 'task_decompose': 0,
    'multi_path_voting': 0, 'memory_read': 0, 'memory_write': 0,
    'timeout_budget': 2, 'early_exit': 0,
    'tool_choice': 'fast_risky', 'diversity_seed': 0,
}

KNOB_DESC = {
    'verify_depth': '结果校验深度 0=不校验 1=只查最终步 2=每步都查 (检测静默错误; 对显性报错族无效)',
    'retries': '失败重试次数 0-8 (对显性报错族强; 消耗预算)',
    'task_decompose': '任务分解 0=整体算 1=两段 2=逐步 (降低多步任务的每步复杂度/错误率)',
    'multi_path_voting': '多路径独立求解后投票 0=关 3=三路 5=五路 取众数 (抗噪声抗偶发错误)',
    'memory_read': '读历史同构任务经验 0/1 (多步任务族中提供先验)',
    'memory_write': '写经验到记忆库 0/1 (与read配合才有收益; 单独开是纯开销)',
    'timeout_budget': 'probe/时间预算 1-8 (部分可观测族需要足够预算探明隐藏信息)',
    'early_exit': '高置信提前退出 0/1 (省下预算转投probe)',
    'tool_choice': '工具选择 fast_risky=快但错率高 slow_safe=慢但稳 both=双路',
    'diversity_seed': '路径多样性种子 0=路径相关(共享错误) 1-3=独立 (让多路投票真正生效)',
}

# 因果杠杆表 (报告对照用; 加粗=强)
LEVER_TABLE = {
    'A': {'strong': ['verify_depth', 'retries'], 'mid': ['multi_path_voting', 'early_exit', 'tool_choice'], 'weak': ['task_decompose', 'memory_read', 'memory_write', 'timeout_budget', 'diversity_seed']},
    'B': {'strong': ['retries'], 'mid': ['timeout_budget', 'tool_choice'], 'weak': ['verify_depth', 'task_decompose', 'multi_path_voting', 'memory_read', 'memory_write', 'early_exit', 'diversity_seed']},
    'C': {'strong': ['task_decompose', 'memory_read', 'memory_write'], 'mid': ['verify_depth', 'retries', 'timeout_budget'], 'weak': ['multi_path_voting', 'early_exit', 'tool_choice', 'diversity_seed']},
    'D': {'strong': ['timeout_budget', 'early_exit'], 'mid': ['verify_depth', 'retries', 'multi_path_voting', 'tool_choice'], 'weak': ['task_decompose', 'memory_read', 'memory_write', 'diversity_seed']},
    'E': {'strong': ['multi_path_voting', 'diversity_seed'], 'mid': ['retries'], 'weak': ['verify_depth', 'task_decompose', 'memory_read', 'memory_write', 'timeout_budget', 'early_exit', 'tool_choice']},
}

FAMILY_DESC = {
    'A': '静默损坏族: 工具调用有50%概率返回错值且不报错, 只有结果校验能发现',
    'B': '显性报错族: 工具调用35%概率失败并明确报错, 重试即可恢复',
    'C': '多步依赖族: 4步串行计算, 前步输出是后步输入, 每步25%逻辑错误率(分解可降低)',
    'D': '部分可观测族: 4步中前2步信息可见, 后2步需probe探明(每probe耗1预算), 盲算错误率45%',
    'E': '噪声标签族: 执行错误率10%, 且校验信号本身5%翻转(误报/漏报), 需多路投票抗噪',
}


def make_tasks(family, n, seed):
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        steps = 4 if family in ('C', 'D') else 2
        tasks.append({'id': '%s%03d' % (family, i), 'family': family, 'steps': steps})
    return tasks


TASKS = {f: make_tasks(f, N_TASKS, BASE_SEED + 100 * ord(f)) for f in FAMILIES}


# ---------------- sandbox ----------------

def _vote(paths):
    ok = sum(1 for p in paths if p[0])
    return (ok * 2 > len(paths), 'vote_' + ('ok' if ok * 2 > len(paths) else 'fail'))


def solve_A(task, cfg, rng, mem):
    steps = 2
    p = 0.5
    if cfg['tool_choice'] in ('slow_safe', 'both'):
        p = 0.25
    p -= 0.04 * cfg['task_decompose']
    vd = cfg['verify_depth']

    def one_path():
        for s in range(steps):
            detected = (vd == 2) or (vd >= 1 and s == steps - 1)
            if detected:
                for a in range(cfg['retries'] + 1):
                    if rng.random() >= p:
                        break
                else:
                    return (False, 'corruption_exhausted_retries_step%d' % s)
            else:
                if rng.random() < p:
                    return (False, 'silent_corruption_accepted_step%d' % s)
        return (True, '')

    k = cfg['multi_path_voting']
    if k > 0:
        if cfg['diversity_seed'] == 0:
            k = 1 + (k - 1) // 2
        return _vote([one_path() for _ in range(k)])
    return one_path()


def solve_B(task, cfg, rng, mem):
    steps = 2
    p = 0.35
    if cfg['tool_choice'] in ('slow_safe', 'both'):
        p = 0.18

    def one_path():
        for s in range(steps):
            for a in range(cfg['retries'] + 1):
                if rng.random() >= p:
                    break
            else:
                return (False, 'error_retry_exhausted_step%d' % s)
        return (True, '')

    k = cfg['multi_path_voting']
    if k > 0:
        if cfg['diversity_seed'] == 0:
            k = 1 + (k - 1) // 2
        return _vote([one_path() for _ in range(k)])
    return one_path()


def solve_C(task, cfg, rng, mem):
    steps = 4
    p = 0.25 - 0.06 * cfg['task_decompose']
    if cfg['memory_read'] and cfg['memory_write']:
        p -= 0.05
    elif cfg['memory_read'] and mem.get('C', 0) >= 10:
        p -= 0.03
    vd = cfg['verify_depth']

    def one_path():
        for s in range(steps):
            if vd >= 1:
                for a in range(cfg['retries'] + 1):
                    if rng.random() >= p:
                        break
                else:
                    return (False, 'logic_error_step%d' % s)
            else:
                if rng.random() < p:
                    return (False, 'logic_error_step%d' % s)
        return (True, '')

    k = cfg['multi_path_voting']
    if k > 0:
        if cfg['diversity_seed'] == 0:
            k = 1 + (k - 1) // 2
        return _vote([one_path() for _ in range(k)])
    return one_path()


def solve_D(task, cfg, rng, mem):
    steps = 4
    tb = cfg['timeout_budget'] + (1 if cfg['early_exit'] else 0)
    probed = min(steps, tb)

    def one_path():
        for s in range(steps):
            step_p = 0.07 if s < probed else 0.45
            if cfg['verify_depth'] >= 1:
                for a in range(cfg['retries'] + 1):
                    if rng.random() >= step_p:
                        break
                else:
                    return (False, ('blind_step_error_step%d' if s >= probed else 'probed_step_error_step%d') % s)
            else:
                if rng.random() < step_p:
                    return (False, ('blind_step_error_step%d' if s >= probed else 'probed_step_error_step%d') % s)
        return (True, '')

    k = cfg['multi_path_voting']
    if k > 0:
        if cfg['diversity_seed'] == 0:
            k = 1 + (k - 1) // 2
        return _vote([one_path() for _ in range(k)])
    return one_path()


def solve_E(task, cfg, rng, mem):
    steps = 2
    p = 0.10
    vd = cfg['verify_depth']

    def one_path():
        for s in range(steps):
            bad = rng.random() < p
            if not bad:
                # 执行对了, 但噪声信号5%概率误报"错了" → 多余重试(纯浪费但通常能恢复)
                if vd >= 1 and rng.random() < 0.05:
                    for a in range(cfg['retries']):
                        if rng.random() >= p:
                            break
                    else:
                        return (False, 'noise_false_alarm_exhausted_step%d' % s)
                continue
            # 执行错了
            if vd >= 1:
                # 信号5%概率漏报 → 接受错误答案
                if rng.random() < 0.05:
                    return (False, 'noise_accepted_wrong_step%d' % s)
                for a in range(cfg['retries']):
                    if rng.random() >= p:
                        break
                else:
                    return (False, 'exec_error_step%d' % s)
            else:
                return (False, 'exec_error_step%d' % s)
        return (True, '')

    k = cfg['multi_path_voting']
    if k > 0:
        if cfg['diversity_seed'] == 0:
            k = 1 + (k - 1) // 2
        return _vote([one_path() for _ in range(k)])
    return one_path()


SOLVERS = {'A': solve_A, 'B': solve_B, 'C': solve_C, 'D': solve_D, 'E': solve_E}


def eval_family(cfg, family, seed, tasks=None):
    tasks = tasks or TASKS[family]
    rng = random.Random(seed + 7 * hash(family) % 10000)
    mem = {'C': 0}
    ok = 0
    fail_reasons = {}
    for t in tasks:
        good, why = SOLVERS[family](t, cfg, rng, mem)
        if good:
            ok += 1
            if family == 'C' and cfg['memory_write']:
                mem['C'] += 1
        else:
            key = re.sub(r'step\d+', 'step*', why)
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
    return ok / float(len(tasks)), fail_reasons


def eval_config(cfg, seed):
    out = {}
    for f in FAMILIES:
        out[f], _ = eval_family(cfg, f, seed)
    out['mean'] = sum(out[f] for f in FAMILIES) / 5.0
    out['cost'] = cfg_cost(cfg)
    out['score'] = round(out['mean'] - out['cost'], 4)
    return out


def normalize_cfg(raw):
    """LLM 输出 -> 合法 config; 非法值夹到域内."""
    cfg = copy.deepcopy(DEFAULT_CFG)
    if not isinstance(raw, dict):
        return None
    changed = {}
    for k, v in raw.items():
        if k not in KNOB_SPEC:
            continue
        dom = KNOB_SPEC[k]
        if isinstance(dom[0], int) and not isinstance(dom[0], bool):
            try:
                v = int(v)
            except Exception:
                continue
            if v not in dom:
                lo, hi = min(dom), max(dom)
                v = max(lo, min(hi, v))
                if v not in dom:  # e.g. voting domain gaps
                    v = min(dom, key=lambda x: abs(x - v))
        elif isinstance(dom[0], str):
            v = str(v)
            if v not in dom:
                continue
        else:
            v = 1 if v else 0
        if cfg[k] != v:
            changed[k] = v
        cfg[k] = v
    if not changed:
        return None, None
    return cfg, changed


# ---------------- LLM proposer ----------------

class Proposer:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, dtype=torch.bfloat16,
                attn_implementation='eager')
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, torch_dtype=torch.bfloat16,
                attn_implementation='eager')
        self.model.to('cuda')
        self.model.eval()

    def gen(self, prompt, temp=0.8):
        import torch
        msgs = [{'role': 'user', 'content': prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=220, do_sample=True, temperature=temp,
                top_p=0.9, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)


def parse_json_loose(s):
    # 1) greedy: 完整嵌套对象 (含 "changes" 外壳)
    m = re.search(r'\{.*\}', s, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 2) fallback: 首个扁平对象, 若无 changes 键则视作 changes 本体
    m2 = re.search(r'\{[^{}]*\}', s, re.S)
    if m2:
        try:
            o = json.loads(m2.group(0))
            if isinstance(o, dict):
                if 'changes' in o:
                    return o
                return {'changes': o}
        except Exception:
            pass
    return None


def build_prompt(family, cfg, sr, fails, history, round_no):
    lines = []
    lines.append('你是自我进化引擎的策略提案器。')
    lines.append('任务族 %s: %s' % (family, FAMILY_DESC[family]))
    lines.append('')
    lines.append('当前配置(10个旋钮):')
    for k in KNOB_SPEC:
        lines.append('  %s = %s  # %s' % (k, cfg[k], KNOB_DESC[k]))
    lines.append('')
    lines.append('当前成功率 sr=%.3f, cost=%.3f, score=%.3f' % (sr, cfg_cost(cfg), score_of(sr, cfg)))
    lines.append('优化目标: 最大化 score = sr - cost (旋钮有代价, 全开会扣分, 要找最小代价达标组合)')
    lines.append(COST_FORMULA)
    if fails:
        total = sum(fails.values())
        lines.append('失败原因分布:')
        for k, v in sorted(fails.items(), key=lambda x: -x[1]):
            lines.append('  %s: %d/%d' % (k, v, total))
    if history:
        lines.append('')
        lines.append('历史提案(按时间序):')
        for h in history[-4:]:
            if h.get('changes') is None:
                lines.append('  R%d: 无有效提案 (%s)' % (h['round'], h.get('reason', '')))
            else:
                lines.append('  R%d: %s -> %s (sr %.3f score %.3f, %s)' % (
                    h['round'], json.dumps(h['changes']),
                    '接受' if h['accepted'] else '拒绝',
                    h.get('sr', -1), h.get('score', -1), h.get('reason', '')))
    lines.append('')
    lines.append('请提出对若干旋钮的修改, 提高该族 score = sr - cost。可以一次改多个旋钮。')
    lines.append('只输出一个JSON对象, 格式:')
    lines.append('{"title": "一句话", "rationale": "为什么", "changes": {"旋钮名": 新值, ...}}')
    return '\n'.join(lines)


# ---------------- experiment ----------------

RESULT = {
    'phase': 'G1_D1_knobs10',
    'knob_spec': KNOB_SPEC,
    'lever_table': LEVER_TABLE,
    'families': {},
    'baselines': {},
}


def log(s):
    print(s, flush=True)


def main():
    t0 = time.time()
    log('==== G1/D1: 10-knob sandbox, 5 families ====')

    # 1) default baseline
    base = eval_config(DEFAULT_CFG, BASE_SEED)
    log('[base] ' + json.dumps({k: round(v, 4) for k, v in base.items()}))
    RESULT['baselines']['default'] = base

    # 2) random search 200 (score = sr - cost)
    rng = random.Random(31337)
    best_rand = None
    rand_trace = []
    for i in range(200):
        cfg = copy.deepcopy(DEFAULT_CFG)
        for k, dom in KNOB_SPEC.items():
            cfg[k] = rng.choice(dom)
        sr = eval_config(cfg, BASE_SEED)
        rand_trace.append({'i': i, 'score': sr['score'], 'mean': round(sr['mean'], 4)})
        if best_rand is None or sr['score'] > best_rand[0]['score']:
            best_rand = (sr, cfg)
    log('[random200] best score=%.4f mean=%.4f cfg=%s' %
        (best_rand[0]['score'], best_rand[0]['mean'], json.dumps(best_rand[1])))
    RESULT['baselines']['random_best'] = {'sr': best_rand[0], 'cfg': best_rand[1]}
    RESULT['baselines']['random_trace_top'] = sorted(rand_trace, key=lambda x: -x['score'])[:10]

    # 3) 2-knob exhaustive (verify_depth {0,1} x retries 0..8, 其余 default)
    best2 = None
    for vd in (0, 1):
        for rt in range(9):
            cfg = copy.deepcopy(DEFAULT_CFG)
            cfg['verify_depth'] = vd
            cfg['retries'] = rt
            sr = eval_config(cfg, BASE_SEED)
            if best2 is None or sr['score'] > best2[0]['score']:
                best2 = (sr, cfg)
    log('[2knob] best score=%.4f mean=%.4f cfg=%s' %
        (best2[0]['score'], best2[0]['mean'], json.dumps({k: best2[1][k] for k in ('verify_depth', 'retries')})))
    RESULT['baselines']['twoknob_best'] = {'sr': best2[0], 'cfg': best2[1]}

    # 4) LLM population evolution per family
    proposer = Proposer()
    log('[llm] proposer loaded %.1fs' % (time.time() - t0))

    for family in FAMILIES:
        log('---- family %s ----' % family)
        cfg = copy.deepcopy(DEFAULT_CFG)
        sr, fails = eval_family(cfg, family, BASE_SEED)
        history = []
        rounds = []
        for rnd in range(1, N_ROUNDS + 1):
            prompt = build_prompt(family, cfg, sr, fails, history, rnd)
            cands = []
            for s_i in range(K_SAMPLES):
                try:
                    raw = proposer.gen(prompt)
                except Exception as e:
                    log('[%s/R%d] gen error %s' % (family, rnd, e))
                    continue
                obj = parse_json_loose(raw)
                if obj is None:
                    cands.append({'raw': raw[-160:], 'valid': False, 'why': 'no_json'})
                    continue
                nc, changed = normalize_cfg(obj.get('changes', {}))
                if nc is None:
                    cands.append({'raw': raw[-160:], 'valid': False, 'why': 'no_op_or_illegal',
                                  'parsed': obj.get('changes')})
                    continue
                cands.append({'raw': raw[-160:], 'valid': True, 'changes': changed,
                              'cfg': nc, 'title': obj.get('title', ''),
                              'rationale': obj.get('rationale', '')})
            # gate prescreen on independent seed (score = gate_sr - cost)
            gate_seed = GATE_SEED + 1000 * ord(family) + rnd
            best_c = None
            for c in cands:
                if not c.get('valid'):
                    continue
                g_sr, _ = eval_family(c['cfg'], family, gate_seed)
                c['gate_sr'] = round(g_sr, 4)
                c['gate_score'] = round(score_of(g_sr, c['cfg']), 4)
                if best_c is None or c['gate_score'] > best_c['gate_score']:
                    best_c = c
            cur_score = score_of(sr, cfg)
            rec = {'round': rnd, 'base_sr': round(sr, 4), 'base_score': round(cur_score, 4),
                   'candidates': cands, 'accepted': None, 'gate_sr': None}
            if best_c is None:
                log('[%s/R%d] no valid candidate, skipped' % (family, rnd))
                history.append({'round': rnd, 'changes': None, 'accepted': False,
                                'reason': 'no valid candidate'})
                rounds.append(rec)
                continue
            c = best_c
            g_sr = c['gate_sr']
            g_score = c['gate_score']
            rec['gate_sr'] = g_sr
            rec['gate_score'] = g_score
            if g_score < cur_score + MIN_GAIN:
                log('[%s/R%d] best gate score=%.3f (sr=%.3f cost=%.3f) < base score %.3f + %.2f, REJECTED (min_gain)' %
                    (family, rnd, g_score, g_sr, cfg_cost(c['cfg']), cur_score, MIN_GAIN))
                history.append({'round': rnd, 'changes': c['changes'], 'accepted': False,
                                'sr': round(g_sr, 4), 'score': g_score,
                                'reason': 'gate score below min_gain'})
                rec['accepted'] = False
                rec['rejected'] = c
                rounds.append(rec)
                continue
            cfg = c['cfg']
            sr = g_sr
            _, fails = eval_family(cfg, family, BASE_SEED)
            log('[%s/R%d] ACCEPTED %s gate sr=%.3f cost=%.3f score=%.3f' %
                (family, rnd, json.dumps(c['changes']), g_sr, cfg_cost(cfg), g_score))
            history.append({'round': rnd, 'changes': c['changes'], 'accepted': True,
                            'sr': round(g_sr, 4), 'score': g_score, 'reason': 'gate pass'})
            rec['accepted'] = True
            rec['accepted_changes'] = c['changes']
            rounds.append(rec)
        fam_state = {'final_cfg': cfg, 'rounds': rounds, 'history': history}
        RESULT['families'][family] = fam_state

    # 5) final harvest on fresh seed, all families, all configs
    log('---- final harvest (fresh seed %d) ----' % FINAL_SEED)
    final = {'default': eval_config(DEFAULT_CFG, FINAL_SEED),
             'random_best': eval_config(best_rand[1], FINAL_SEED),
             'twoknob_best': eval_config(best2[1], FINAL_SEED)}
    for family in FAMILIES:
        final['evo_%s' % family] = eval_config(RESULT['families'][family]['final_cfg'], FINAL_SEED)
    RESULT['final'] = {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in final.items()}

    # 每族单独进化 (单族 config 在本族的 score) vs 基线
    verdict = {}
    for family in FAMILIES:
        def fam_score(entry, fam):
            return entry[fam] - entry['cost']
        evo_sr = fam_score(final['evo_%s' % family], family)
        rb = fam_score(final['random_best'], family)
        tk = fam_score(final['twoknob_best'], family)
        df = fam_score(final['default'], family)
        verdict[family] = {
            'evo_score': round(evo_sr, 4), 'random_best_score': round(rb, 4),
            'twoknob_best_score': round(tk, 4), 'default_score': round(df, 4),
            'evo_sr': round(final['evo_%s' % family][family], 4),
            'evo_cost': final['evo_%s' % family]['cost'],
            'pass_rand': evo_sr > rb, 'pass_2knob': evo_sr > tk,
        }
        log('[verdict/%s] %s' % (family, json.dumps(verdict[family])))
    RESULT['verdict'] = verdict

    n_pass = sum(1 for f in FAMILIES
                 if verdict[f]['pass_rand'] and verdict[f]['pass_2knob'])
    RESULT['g1_gate'] = {'families_passed': n_pass, 'total': len(FAMILIES),
                         'gate_pass': n_pass >= 3, 'metric': 'score=sr-cost'}
    log('==== G1 GATE (score=sr-cost): %d/5 families passed both baselines -> %s ====' %
        (n_pass, 'PASS' if n_pass >= 3 else 'FAIL'))
    log('total %.1fs' % (time.time() - t0))

    with open('/root/private_data/phase_d1_knobs.json', 'w') as f:
        json.dump(RESULT, f, ensure_ascii=False, indent=1)
    log('saved /root/private_data/phase_d1_knobs.json')
    log('D1_DONE')


if __name__ == '__main__':
    main()
