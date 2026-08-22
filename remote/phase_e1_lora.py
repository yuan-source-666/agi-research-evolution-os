#!/usr/bin/env python3
"""Phase E1: 进化档案回流 LoRA 微调提议器（Level 2→3 正门）。

流程（全部真实数据，无合成沙箱）：
1. 收集 D2 档案（/root/private_data/phase_d2_real.json 里所有带完整字段的策略+真实 acc）
2. 用基线提议器再采样 12 个策略，真实 GSM8K dev 评估 → 扩充档案（STaR 式拒绝采样）
3. 高分策略构造 SFT 对：prompt=D2 同款提议器模板，completion=策略 JSON
4. peft LoRA 微调 Qwen2.5-1.5B（手写训练循环，避开 trl API 版本坑）
5. 微调后提议器出 6 个策略，dev 真实评估
门禁 E1: 微调后提议器生成策略的 best/mean 同时 > 基线提议器生成策略的 best/mean
产物: /root/private_data/phase_e1_lora.json + adapter /root/private_data/e1_adapter
"""
import os, json, re, time, random, urllib.request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = '/root/private_data/Qwen2.5-1.5B-Instruct'
DATA_URL = ('https://raw.githubusercontent.com/openai/grade-school-math/'
            'master/grade_school_math/data/test.jsonl')
DATA_PATH = '/root/private_data/gsm8k_test.jsonl'
D2_PATH = '/root/private_data/phase_d2_real.json'
OUT_PATH = '/root/private_data/phase_e1_lora.json'
ADAPTER_PATH = '/root/private_data/e1_adapter'
N_DEV = 100
BATCH = 20
MAX_NEW = 512
N_EXPAND = 12      # 基线提议器扩展采样数
N_EVAL_FT = 6      # 微调后提议器评估数
GOOD_ACC = 0.30    # 正例阈值（≈直答水平）
random.seed(7)
RESULT = {'stage': 'init', 'ts_start': time.strftime('%F %T')}


def save():
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(RESULT, f, ensure_ascii=False, indent=1)


def load_data():
    """与 D2 完全一致的数据加载（random.seed 不同会切分不同，这里固定取 dev 用）。"""
    if not os.path.exists(DATA_PATH):
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
    return rows[:N_DEV]


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
    return [tok.decode(ids[enc['input_ids'].shape[1]:],
                       skip_special_tokens=True) for i, ids in enumerate(out)]


def extract_num(text):
    if not isinstance(text, str) or not text:
        return None
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
    return max(set(nums), key=nums.count) if nums else None


# ---------------- 策略执行（与 D2 字面一致） ----------------
def build_msgs(strat, tasks):
    msgs = []
    for t in tasks:
        q = t['question']
        if strat.get('decompose'):
            q += ('\n\nFirst break the problem into 2-3 smaller sub-questions, '
                  'answer each one, then combine them for the final answer.')
        msgs.append([{'role': 'user',
                      'content': ('%s\n\nProblem: %s\n\nEnd your reply '
                                  'with "#### <final numeric answer>".'
                                  % (strat['instruction'], q))}])
    return msgs


def run_strategy(strat, tasks):
    msgs = build_msgs(strat, tasks)
    preds = chat_batch(msgs, temperature=0)
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
    correct = sum(
        1 for t, p in zip(tasks, preds)
        if (p if isinstance(p, float) else extract_num(p)) is not None
        and abs((p if isinstance(p, float) else extract_num(p)) - t['gold']) < 1e-4)
    return correct / len(tasks), time.time() - t0


# ---------------- 提议器（与 D2 字面一致） ----------------
PROPOSER_TMPL = (
    'You are a strategy optimizer for a math-solving system. '
    'Current strategies and their accuracies on a real GSM8K dev set:\n%s\n\n'
    'Best so far: %s (acc %.3f).\n\n'
    'Propose %d NEW strategies as a JSON array. Each element: '
    '{"name": str, "instruction": str (how to approach and reason about the '
    'problem, be specific and creative, 1-3 sentences), '
    '"decompose": bool, "verify": bool, "vote_k": 0 or 3}. '
    'Reply with ONLY the JSON array.')


def propose(history, best, best_acc, temperature=0.9):
    prompt = PROPOSER_TMPL % (
        json.dumps([{'name': h['name'], 'acc': round(h['acc'], 3)}
                    for h in history[-8:]], ensure_ascii=False),
        json.dumps({k: best[k] for k in
                    ('name', 'instruction', 'decompose', 'verify', 'vote_k')},
                   ensure_ascii=False), best_acc, 3)
    raw = chat_batch([[{'role': 'user', 'content': prompt}]],
                     temperature=temperature, max_new=600)[0]
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
    return out[:3]


def strat_json(s):
    return json.dumps([{k: s[k] for k in
                        ('name', 'instruction', 'decompose',
                         'verify', 'vote_k')}], ensure_ascii=False)


# ---------------- 主流程 ----------------
def main():
    dev = load_data()
    print('dev set: %d real GSM8K problems' % len(dev))

    # ---- 1. 收集 D2 档案 ----
    d2 = json.load(open(D2_PATH, encoding='utf-8'))
    archive = []
    for s in d2.get('seeds', []):
        archive.append({k: s[k] for k in
                        ('name', 'instruction', 'decompose',
                         'verify', 'vote_k', 'acc')})
    for key in ('best_dev', 'best_evolved_dev'):
        s = d2.get(key)
        if s:
            archive.append({k: s[k] for k in
                            ('name', 'instruction', 'decompose',
                             'verify', 'vote_k', 'acc')})
    # 去重
    seen, arch = set(), []
    for s in archive:
        if s['name'] not in seen:
            seen.add(s['name'])
            arch.append(s)
    print('D2 archive: %d full strategies' % len(arch))
    RESULT['stage'] = 'd2_archive'
    RESULT['d2_archive'] = arch
    save()

    # ---- 2. 基线提议器扩展采样（真实评估，STaR 式） ----
    print('== baseline proposer expansion: %d samples ==' % N_EXPAND)
    RESULT['stage'] = 'base_expand'
    base_gen = []
    for i in range(N_EXPAND):
        ctx = random.sample(arch, min(6, len(arch)))
        best = max(ctx, key=lambda s: s['acc'])
        cands = propose(ctx, best, best['acc'])
        for c in cands[:1]:  # 每次采样取 1 个，控制评估时间
            acc, dt = evaluate(c, dev)
            c['acc'] = acc
            arch.append(c)
            base_gen.append(c)
            print('  base%-2d %-24s acc=%.3f  (%.0fs)' % (i, c['name'], acc, dt))
            RESULT['base_gen'] = base_gen
            save()
    base_best = max((s['acc'] for s in base_gen), default=0.0)
    base_mean = (sum(s['acc'] for s in base_gen) / len(base_gen)
                 if base_gen else 0.0)
    print('baseline proposer: best=%.3f mean=%.3f' % (base_best, base_mean))

    # ---- 3. 构造 SFT 对 ----
    good = [s for s in arch if s['acc'] >= GOOD_ACC]
    if len(good) < 4:  # 兜底：取 top 6
        good = sorted(arch, key=lambda s: -s['acc'])[:6]
    print('== SFT pairs: %d good strategies ==' % len(good))
    pairs = []
    for g in good:
        for _ in range(3):  # 每个正例 3 个不同上下文
            ctx = random.sample(arch, min(6, len(arch)))
            if g in ctx:
                ctx = [s for s in ctx if s is not g][:6] or arch[:2]
            best = max(ctx, key=lambda s: s['acc'])
            if best['acc'] < g['acc']:
                best = g  # 保证目标策略是"更优提议"
            prompt = PROPOSER_TMPL % (
                json.dumps([{'name': h['name'], 'acc': round(h['acc'], 3)}
                            for h in ctx], ensure_ascii=False),
                json.dumps({k: best[k] for k in
                            ('name', 'instruction', 'decompose',
                             'verify', 'vote_k')}, ensure_ascii=False),
                best['acc'], 3)
            pairs.append({'prompt': prompt, 'completion': strat_json(g)})
    print('total SFT pairs: %d' % len(pairs))

    # ---- 4. LoRA 微调（手写循环） ----
    RESULT['stage'] = 'lora_train'
    RESULT['sft_pairs'] = len(pairs)
    save()
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                      'gate_proj', 'up_proj', 'down_proj'],
                      task_type='CAUSAL_LM')
    pm = get_peft_model(model, lcfg)
    pm.print_trainable_parameters()
    pm.train()

    def build_example(p):
        user_text = tok.apply_chat_template(
            [{'role': 'user', 'content': p['prompt']}],
            tokenize=False, add_generation_prompt=True)
        full_text = user_text + p['completion'] + tok.eos_token
        ids = tok(full_text, return_tensors='pt', add_special_tokens=False
                  ).input_ids[0]
        n_prompt = len(tok(user_text, add_special_tokens=False).input_ids)
        labels = ids.clone()
        labels[:n_prompt] = -100
        return ids, labels

    examples = [build_example(p) for p in pairs]
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad],
                            lr=2e-4)
    EPOCHS, BS = 3, 2
    losses = []
    for ep in range(EPOCHS):
        random.shuffle(examples)
        tot, nb = 0.0, 0
        for i in range(0, len(examples), BS):
            batch = examples[i:i + BS]
            L = max(len(x[0]) for x in batch)
            input_ids = torch.full((len(batch), L), tok.pad_token_id,
                                   dtype=torch.long)
            lab = torch.full((len(batch), L), -100, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (ids, labels) in enumerate(batch):
                input_ids[j, -len(ids):] = ids   # 左 padding（与推理一致）
                lab[j, -len(labels):] = labels
                att[j, -len(ids):] = 1
            out = pm(input_ids=input_ids.to('cuda'),
                     attention_mask=att.to('cuda'),
                     labels=lab.to('cuda'))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            tot += out.loss.item()
            nb += 1
        print('epoch %d: loss %.4f' % (ep, tot / max(nb, 1)))
        losses.append(tot / max(nb, 1))
        RESULT['train_loss'] = losses
        save()
    pm.eval()
    pm.save_pretrained(ADAPTER_PATH)
    print('adapter saved:', ADAPTER_PATH)

    # ---- 5. 微调后提议器评估 ----
    print('== fine-tuned proposer: %d strategies ==' % N_EVAL_FT)
    RESULT['stage'] = 'ft_eval'
    save()
    ft_gen = []
    ctx = random.sample(arch, 6)
    best = max(ctx, key=lambda s: s['acc'])
    got = 0
    while got < N_EVAL_FT:
        cands = propose(ctx, best, best['acc'])
        for c in cands:
            if got >= N_EVAL_FT:
                break
            acc, dt = evaluate(c, dev)
            c['acc'] = acc
            arch.append(c)
            ft_gen.append(c)
            got += 1
            print('  ft%-2d %-24s acc=%.3f  (%.0fs)' % (got, c['name'], acc, dt))
            RESULT['ft_gen'] = ft_gen
            save()
        ctx = random.sample(arch, 6)
        best = max(ctx, key=lambda s: s['acc'])
    ft_best = max(s['acc'] for s in ft_gen)
    ft_mean = sum(s['acc'] for s in ft_gen) / len(ft_gen)

    # ---- 6. 门禁判定 ----
    seed_best = max((s['acc'] for s in d2.get('seeds', [])), default=0.0)
    gate = (ft_best > base_best + 0.02) and (ft_mean > base_mean)
    RESULT.update({
        'stage': 'done',
        'summary': {
            'base_proposer': {'best': round(base_best, 4),
                              'mean': round(base_mean, 4),
                              'n': len(base_gen)},
            'ft_proposer': {'best': round(ft_best, 4),
                            'mean': round(ft_mean, 4),
                            'n': len(ft_gen)},
            'seed_best': round(seed_best, 4),
            'n_good': len(good), 'n_pairs': len(pairs)},
        'e1_gate': bool(gate),
        'ts_end': time.strftime('%F %T')})
    save()
    print('BASE  best=%.3f mean=%.3f' % (base_best, base_mean))
    print('FT    best=%.3f mean=%.3f' % (ft_best, ft_mean))
    print('SEED  best=%.3f' % seed_best)
    print('E1 GATE:', 'PASS' if gate else 'FAIL')

    # 老板规矩：跑完立即卸载显存
    del pm, model, opt
    torch.cuda.empty_cache()
    print('gpu released')


if __name__ == '__main__':
    main()
