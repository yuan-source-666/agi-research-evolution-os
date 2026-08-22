#!/usr/bin/env python3
"""Phase D2: 真实任务上的策略程序进化（GSM8K）。

老板指令（2026-08-22 12:08）：合成沙箱冻结，直接上真任务主线。
- 任务：GSM8K test 真实数学题（openai/grade-school-math 官方数据集）
- 进化对象：策略程序 = {自由文本 instruction + 控制流(decompose/verify/vote_k)}
- 门禁：holdout 上进化最优 > 最佳手写种子策略 => D2 PASS
- 产物：/root/private_data/phase_d2_real.json（含 E1 微调档案）
"""
import os, json, re, time, random, urllib.request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = '/root/private_data/Qwen2.5-1.5B-Instruct'
DATA_URL = ('https://raw.githubusercontent.com/openai/grade-school-math/'
            'master/grade_school_math/data/test.jsonl')
DATA_PATH = '/root/private_data/gsm8k_test.jsonl'
OUT_PATH = '/root/private_data/phase_d2_real.json'
N_DEV, N_HOLD = 100, 50
N_EVO_ROUNDS, N_CAND = 3, 3
BATCH = 20
MAX_NEW = 512
random.seed(0)
RESULT = {'stage': 'init', 'ts_start': time.strftime('%F %T')}


def save():
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(RESULT, f, ensure_ascii=False, indent=1)


# ---------------- 数据 ----------------
def load_data():
    if not os.path.exists(DATA_PATH):
        print('downloading GSM8K test set ...')
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    rows = []
    for line in open(DATA_PATH, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        m = re.search(r'####\s*([\d,\.\-]+)', r.get('answer', ''))
        if not m:
            continue
        try:
            r['gold'] = float(m.group(1).replace(',', ''))
        except ValueError:
            continue
        rows.append(r)
    random.shuffle(rows)
    print('loaded %d usable problems' % len(rows))
    return rows[:N_DEV], rows[N_DEV:N_DEV + N_HOLD]


# ---------------- 模型 ----------------
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda')
model.eval()
print('model loaded, gpu mem %.1f GB' % (torch.cuda.memory_allocated() / 1e9))


@torch.no_grad()
def chat_batch(msgs_list, temperature=0.3, max_new=MAX_NEW):
    texts = [tok.apply_chat_template(m, tokenize=False,
                                     add_generation_prompt=True)
             for m in msgs_list]
    enc = tok(texts, return_tensors='pt', padding=True,
              add_special_tokens=False).to('cuda')
    out = model.generate(
        **enc, max_new_tokens=max_new,
        do_sample=temperature > 0, temperature=max(temperature, 0.05),
        top_p=0.9, pad_token_id=tok.eos_token_id)
    res = []
    for i, ids in enumerate(out):
        res.append(tok.decode(ids[enc['input_ids'].shape[1]:],
                              skip_special_tokens=True))
    return res


def extract_num(text):
    m = re.findall(r'####\s*\$?([\d,\.\-]+)', text)
    if not m:
        m = re.findall(r'[Aa]nswer[:\s]*\S{0,3}\$?([\d,\.\-]+)', text)
    if not m:
        m = re.findall(r'-?\$?[\d,]+(?:\.\d+)?', text)
    if not m:
        return None
    try:
        return float(m[-1].replace(',', '').replace('$', ''))
    except ValueError:
        return None


def majority(nums):
    nums = [n for n in nums if n is not None]
    if not nums:
        return None
    return max(set(nums), key=nums.count)


# ---------------- 策略程序执行 ----------------
def build_msgs(strat, tasks):
    msgs = []
    for t in tasks:
        q = t['question']
        if strat.get('decompose'):
            q += ('\n\nFirst break the problem into 2-3 smaller sub-questions, '
                  'answer each one, then combine them for the final answer.')
        instr = strat['instruction']
        msgs.append([{'role': 'user',
                      'content': ('%s\n\nProblem: %s\n\nEnd your reply '
                                  'with "#### <final numeric answer>".'
                                  % (instr, q))}])
    return msgs


def run_strategy(strat, tasks):
    msgs = build_msgs(strat, tasks)
    preds = chat_batch(msgs, temperature=0)  # 贪心解码：评估必须确定性
    if strat.get('verify'):
        vmsgs = [[{'role': 'user', 'content': (
            'Problem: %s\n\nProposed solution:\n%s\n\nCarefully check the '
            'solution step by step. If you find an error, write out the '
            'corrected full solution. End with "#### <final numeric answer>".'
            % (t['question'], p))}] for t, p in zip(tasks, preds)]
        preds = chat_batch(vmsgs, temperature=0)
    k = int(strat.get('vote_k') or 0)
    if k >= 3:
        all_preds = [preds] + [chat_batch(msgs, temperature=0.8)
                               for _ in range(k - 1)]
        preds = [majority([extract_num(ap[i]) for ap in all_preds])
                 for i in range(len(tasks))]
    return preds


def evaluate(strat, tasks):
    t0 = time.time()
    preds = run_strategy(strat, tasks)
    correct = 0
    traces = []
    for t, p in zip(tasks, preds):
        n = extract_num(p) if not isinstance(p, float) else p
        ok = n is not None and abs(n - t['gold']) < 1e-4
        correct += ok
        traces.append({'ok': ok})
    return correct / len(tasks), time.time() - t0, traces


# ---------------- 种子策略 ----------------
SEEDS = [
    {'name': 'S0_direct', 'instruction':
        'Solve the math problem. Be concise.',
     'decompose': False, 'verify': False, 'vote_k': 0},
    {'name': 'S1_cot', 'instruction':
        'Solve the problem step by step, showing each calculation clearly '
        'before stating the final answer.',
     'decompose': False, 'verify': False, 'vote_k': 0},
    {'name': 'S2_cot_verify', 'instruction':
        'Solve the problem step by step, showing each calculation clearly '
        'before stating the final answer.',
     'decompose': False, 'verify': True, 'vote_k': 0},
    {'name': 'S3_decompose_cot', 'instruction':
        'Solve the problem step by step, showing each calculation clearly.',
     'decompose': True, 'verify': False, 'vote_k': 0},
]

# ---------------- LLM 提议器 ----------------
PROPOSER_TMPL = (
    'You are a strategy optimizer for a math-solving system. '
    'Current strategies and their accuracies on a real GSM8K dev set:\n%s\n\n'
    'Best so far: %s (acc %.3f).\n\n'
    'Propose %d NEW strategies as a JSON array. Each element: '
    '{"name": str, "instruction": str (how to approach and reason about the '
    'problem, be specific and creative, 1-3 sentences), '
    '"decompose": bool, "verify": bool, "vote_k": 0 or 3}. '
    'Reply with ONLY the JSON array.')


def propose(history, best, best_acc):
    prompt = PROPOSER_TMPL % (
        json.dumps([{'name': h['name'], 'acc': round(h['acc'], 3)}
                    for h in history[-8:]], ensure_ascii=False),
        json.dumps(best, ensure_ascii=False), best_acc, N_CAND)
    raw = chat_batch([[{'role': 'user', 'content': prompt}]],
                     temperature=0.9, max_new=600)[0]
    m = re.search(r'\[.*\]', raw, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for c in arr:
        if not isinstance(c, dict) or 'instruction' not in c:
            continue
        out.append({'name': str(c.get('name', 'cand'))[:40],
                    'instruction': str(c['instruction'])[:500],
                    'decompose': bool(c.get('decompose', False)),
                    'verify': bool(c.get('verify', False)),
                    'vote_k': int(c.get('vote_k') or 0)})
    return out[:N_CAND]


# ---------------- 主流程 ----------------
def main():
    dev, hold = load_data()
    RESULT['stage'] = 'seed_eval'
    save()

    print('== round 0: seed strategies on dev(%d) ==' % len(dev))
    history = []
    for s in SEEDS:
        acc, dt, _ = evaluate(s, dev)
        s['acc'] = acc
        history.append(s)
        print('  %-16s acc=%.3f  (%.0fs)' % (s['name'], acc, dt))
        RESULT['seeds'] = history
        save()
    best = max(history, key=lambda s: s['acc'])
    print('best seed:', best['name'], round(best['acc'], 3))
    seed_best = best  # 最佳手写种子（holdout 对手）

    print('== evolution: %d rounds x %d candidates ==' % (N_EVO_ROUNDS, N_CAND))
    evo_log = []
    best_evolved = None  # 只统计 LLM 生成的策略，种子不算（修对比 bug）
    for r in range(N_EVO_ROUNDS):
        RESULT['stage'] = 'evo_round_%d' % r
        save()
        cands = propose(history, best, best['acc'])
        print('round %d: proposer returned %d candidates' % (r, len(cands)))
        for c in cands:
            acc, dt, _ = evaluate(c, dev)
            c['acc'] = acc
            history.append(c)
            evo_log.append({'round': r, 'name': c['name'], 'acc': acc})
            print('  cand %-24s acc=%.3f  (%.0fs)' % (c['name'], acc, dt))
            if best_evolved is None or acc > best_evolved['acc']:
                best_evolved = c
            if acc > best['acc']:
                best = c
                print('  ^ new best')
        RESULT['evo_log'] = evo_log
        RESULT['best_dev'] = {k: best[k] for k in
                              ('name', 'instruction', 'decompose',
                               'verify', 'vote_k', 'acc')}
        RESULT['best_evolved_dev'] = ({k: best_evolved[k] for k in
                                       ('name', 'instruction', 'decompose',
                                        'verify', 'vote_k', 'acc')}
                                      if best_evolved else None)
        save()

    print('== holdout(%d): best evolved vs best seed ==' % len(hold))
    RESULT['stage'] = 'holdout'
    save()
    acc_seed, _, _ = evaluate(seed_best, hold)
    acc_base, _, _ = evaluate(SEEDS[0], hold)
    if best_evolved is not None:
        acc_evo, _, _ = evaluate(best_evolved, hold)
        gate = acc_evo > acc_seed
    else:
        acc_evo, gate = None, False
    RESULT.update({
        'stage': 'done',
        'holdout': {'best_evolved': round(acc_evo, 4) if acc_evo is not None else None,
                    'best_seed': round(acc_seed, 4),
                    'seed_name': seed_best['name'],
                    'evo_name': best_evolved['name'] if best_evolved else None,
                    'baseline_direct': round(acc_base, 4)},
        'd2_gate': bool(gate),
        'ts_end': time.strftime('%F %T')})
    save()
    print('HOLDOUT  evolved(%s)=%s  seed(%s)=%.3f  direct=%.3f'
          % (best_evolved['name'] if best_evolved else 'none',
             ('%.3f' % acc_evo) if acc_evo is not None else 'n/a',
             seed_best['name'], acc_seed, acc_base))
    print('D2 GATE:', 'PASS' if gate else 'FAIL')

    # 老板规矩：跑完立即卸载显存
    torch.cuda.empty_cache()
    print('gpu released')


if __name__ == '__main__':
    main()
