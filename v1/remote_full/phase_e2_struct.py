#!/usr/bin/env python3
"""Phase E2-STRUCT: 结构自进化（结构+权重同时演化）。

进化对象不是提示词、不是配置字典，是模型本身的结构：
  genome = {
    depth:       保留多少层 transformer（深度手术，直接改计算图）
    targets:     LoRA 接到哪些投影层（容量加在哪里，接线结构）
    rank:        每个 LoRA 的瓶颈宽度（结构容量）
  }
每个候选：加载基座 → 砍层 → 接 LoRA → 真实 GSM8K 训练数据等预算 SFT → 真实 dev 评估 → 适应度。

门禁（等预算 A/B，直接验证"结构是智能的一大来源"）：
  进化出的结构在同训练预算下 dev 和 holdout 都 > 基线结构（全 28 层 + 标准 q/v LoRA r16）。

输出: /root/private_data/phase_e2_struct.json
"""
import os, json, time, random, gc, re, urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_PATH = '/root/private_data/Qwen2.5-1.5B-Instruct'
OUT = '/root/private_data/phase_e2_struct.json'
ALL_T = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
         'gate_proj', 'up_proj', 'down_proj']

N_TRAIN = 500      # GSM8K train 真实训练数据
STEPS = 150        # 每个候选同等训练预算
BS = 4
LR = 1e-4
DEV_N = 100
HOLD_N = 50
MAX_NEW = 512

RESULT = {'stage': 'init', 'ts_start': time.strftime('%F %T')}


def save():
    json.dump(RESULT, open(OUT, 'w'), ensure_ascii=False, indent=1)


def log(*a):
    print(*a, flush=True)


# ---------------- 数据（全部真实外部数据集，urllib 直下） ----------------
log('== load GSM8K (real external dataset) ==')
GH = ('https://raw.githubusercontent.com/openai/grade-school-math/'
      'master/grade_school_math/data/')


def load_jsonl(path, url):
    if not os.path.exists(path):
        log('downloading %s ...' % path)
        urllib.request.urlretrieve(url, path)
    rows = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


train = load_jsonl('/root/private_data/gsm8k_train.jsonl', GH + 'train.jsonl')[:N_TRAIN]
test_full = load_jsonl('/root/private_data/gsm8k_test.jsonl', GH + 'test.jsonl')
dev = test_full[:DEV_N]
hold = test_full[DEV_N:DEV_N + HOLD_N]
log('train=%d dev=%d hold=%d' % (len(train), len(dev), len(hold)))

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def build_sft(ex):
    full = tok.apply_chat_template(
        [{'role': 'user', 'content': ex['question']},
         {'role': 'assistant', 'content': ex['answer']}], tokenize=False)
    prompt = tok.apply_chat_template(
        [{'role': 'user', 'content': ex['question']}],
        tokenize=False, add_generation_prompt=True)
    return prompt, full


SFT = [build_sft(ex) for ex in train]
log('sft pairs: %d' % len(SFT))


def extract_num(s):
    m = re.findall(r'####\s*([\d,\.\-]+)', s)
    if not m:
        return None
    try:
        return float(m[-1].replace(',', '').strip().rstrip('.'))
    except Exception:
        return None


@torch.no_grad()
def evaluate(model, tasks):
    model.eval()
    correct = 0
    for t in tasks:
        msgs = [{'role': 'user', 'content':
                 t['question'] +
                 "\n\nReasoning step by step, end with \"#### <final numeric answer>\"."}]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors='pt').to('cuda')
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        gold, pred = extract_num(t['answer']), extract_num(text)
        if gold is not None and pred is not None and abs(pred - gold) < 1e-4:
            correct += 1
    return correct / len(tasks)


# ---------------- 候选：结构手术 + 等预算权重训练 ----------------
def build_candidate(genome):
    """结构手术：砍层（改深度/计算图）+ LoRA 接线（容量加在哪）。"""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation='eager')
    k = genome['depth']
    model.model.layers = model.model.layers[:k]      # 深度手术
    model.config.num_hidden_layers = k
    lcfg = LoraConfig(r=genome['rank'], lora_alpha=genome['rank'] * 2,
                      lora_dropout=0.05, target_modules=genome['targets'],
                      task_type='CAUSAL_LM')
    model = get_peft_model(model, lcfg)              # 接线手术
    model.to('cuda')
    return model


def train_equal_budget(model):
    """每个候选完全相同的训练预算：150 步，lr 1e-4，真实 GSM8K train。"""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    rng = random.Random(0)
    idx = list(range(len(SFT)))
    rng.shuffle(idx)
    step, losses = 0, []
    while step < STEPS:
        for b0 in range(0, len(idx), BS):
            if step >= STEPS:
                break
            batch = [SFT[i] for i in idx[b0:b0 + BS]]
            prompts = [p for p, _ in batch]
            enc = tok([f for _, f in batch], return_tensors='pt', padding=True,
                      truncation=True, max_length=768).to('cuda')
            labels = enc['input_ids'].clone()
            for j, p in enumerate(prompts):
                pl = len(tok(p, add_special_tokens=False)['input_ids'])
                labels[j, :min(pl, labels.shape[1])] = -100
            labels[enc['attention_mask'] == 0] = -100
            out = model(**enc, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(out.loss.item())
            step += 1
    return sum(losses[-20:]) / max(1, len(losses[-20:]))


def run_candidate(genome, tag):
    log('-- candidate %s: %s' % (tag, json.dumps(genome)))
    t0 = time.time()
    model = build_candidate(genome)
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss = train_equal_budget(model)
    acc = evaluate(model, dev)
    log('   loss=%.3f  dev_acc=%.3f  trainable=%.1fM  (%.0fs)'
        % (loss, acc, n_train_p / 1e6, time.time() - t0))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {'tag': tag, 'genome': genome, 'train_loss': round(loss, 4),
            'dev_acc': round(acc, 4), 'trainable_M': round(n_train_p / 1e6, 2),
            'sec': int(time.time() - t0)}


# ---------------- 进化循环 ----------------
rng = random.Random(42)
DEPTHS = [12, 16, 20, 24, 28]
RANKS = [4, 8, 16, 32]


def rand_genome():
    t = [m for m in ALL_T if rng.random() < 0.5] or ['q_proj', 'v_proj']
    return {'depth': rng.choice(DEPTHS), 'targets': sorted(t),
            'rank': rng.choice(RANKS)}


def mutate(g):
    m = dict(g)
    m['targets'] = list(g['targets'])
    if rng.random() < 0.5:
        m['depth'] = rng.choice(DEPTHS)
    if rng.random() < 0.5:
        t = set(m['targets'])
        if t and rng.random() < 0.5:
            t.discard(rng.choice(sorted(t)))
        else:
            t.add(rng.choice(ALL_T))
        m['targets'] = sorted(t) or ['q_proj']
    if rng.random() < 0.3:
        m['rank'] = rng.choice(RANKS)
    return m


log('== baseline structure: full 28 layers + standard q/v LoRA r16 ==')
RESULT['stage'] = 'baseline'
save()
baseline_g = {'depth': 28, 'targets': ['q_proj', 'v_proj'], 'rank': 16}
baseline = run_candidate(baseline_g, 'baseline')
RESULT['baseline'] = baseline
save()

log('== evolution round 1: 3 random structures ==')
RESULT['stage'] = 'evo_r1'
save()
pop = [baseline]
for i in range(3):
    g = rand_genome()
    pop.append(run_candidate(g, 'r1_rand%d' % i))
    RESULT['pop_r1'] = pop
    save()

best = max(pop, key=lambda c: c['dev_acc'])
log('best after r1: %s dev=%.3f' % (best['tag'], best['dev_acc']))

log('== evolution round 2: 2 mutations of best ==')
RESULT['stage'] = 'evo_r2'
save()
for i in range(2):
    g = mutate(best['genome'])
    pop.append(run_candidate(g, 'r2_mut%d' % i))
    RESULT['pop_r2'] = pop
    save()

best = max(pop, key=lambda c: c['dev_acc'])

log('== holdout: best evolved structure vs baseline structure ==')
RESULT['stage'] = 'holdout'
save()
m_b = build_candidate(best['genome'])
h_best = evaluate(m_b, hold)
del m_b; gc.collect(); torch.cuda.empty_cache()
m_base = build_candidate(baseline_g)
h_base = evaluate(m_base, hold)
del m_base; gc.collect(); torch.cuda.empty_cache()

gate = (best['dev_acc'] > baseline['dev_acc']) and (h_best > h_base)
RESULT.update({
    'stage': 'done',
    'best': best,
    'holdout': {'best_struct': round(h_best, 4),
                'baseline_struct': round(h_base, 4)},
    'e2_struct_gate': bool(gate),
    'ts_end': time.strftime('%F %T')})
save()
log('HOLDOUT best_struct=%.3f baseline_struct=%.3f' % (h_best, h_base))
log('E2 STRUCT GATE:', 'PASS' if gate else 'FAIL')
log('all candidates:')
for c in pop:
    log('  %-12s depth=%-2d rank=%-2d targets=%-40s dev=%.3f'
        % (c['tag'], c['genome']['depth'], c['genome']['rank'],
           ','.join(c['genome']['targets']), c['dev_acc']))

gc.collect()
torch.cuda.empty_cache()
log('gpu released')
