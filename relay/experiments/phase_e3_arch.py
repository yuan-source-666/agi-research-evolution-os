#!/usr/bin/env python3
"""E3-ARCH: 从零开始的架构自进化（真架构，不是在 Qwen 上动刀）

进化对象 = 架构本身：
- 基因组 = 异构模块 DAG：每层类型（注意力 / 双向GRU / 前馈 / 门控）+ 跳跃接线（skip 连到哪层）
- 拓扑跨代进化（加块/删块/换类型/改接线），每个架构候选的权重从零训练
- 真实外部数据：AG News 新闻四分类（12万条真实新闻，非合成）
- 证明目标：等预算下，进化出的架构 vs 人类默认固定架构（纯MLP/纯GRU/纯注意力/混合）
门禁：进化最优架构 test 准确率 > 所有固定架构
"""
import os, json, time, random, gc, csv, urllib.request
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

RESULT = {'stage': 'init', 'ts_start': time.strftime('%F %T')}


def save():
    json.dump(RESULT, open('/root/private_data/phase_e3_arch.json', 'w'),
              indent=1, ensure_ascii=False)


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

D = 128; VOCAB = 30000; MAXLEN = 64
BS = 64; STEPS = 500; LR = 3e-4
POP = 8; GENS = 4; KEEP = 4
N_TRAIN = 20000; DEV_N = 2000; TEST_N = 4000

# ---------------- 真实外部数据：AG News ----------------
def robust_download(url, dst, tries=6):
    # 多镜像源：国内 SCNet 直连 GitHub 不稳，jsdelivr/ghfast 镜像可达
    if 'raw.githubusercontent.com' in url:
        path = url.split('raw.githubusercontent.com/', 1)[1]
        owner, repo, ref = path.split('/', 2)
        mirrors = [
            'https://cdn.jsdelivr.net/gh/%s@%s' % (owner + '/' + repo, ref),
            'https://ghfast.top/' + url,
            url,
        ]
    else:
        mirrors = [url]
    per = max(1, tries // len(mirrors))
    for mu in mirrors:
        for i in range(per):
            try:
                tmp = dst + '.part'
                req = urllib.request.Request(mu, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=180) as r, open(tmp, 'wb') as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                if os.path.getsize(tmp) > 10_000_000:
                    os.replace(tmp, dst)
                    log('download ok from %s' % mu)
                    return
                log('download try %d (%s): file too small (%d bytes), retry'
                    % (i + 1, mu, os.path.getsize(tmp)))
            except Exception as e:
                log('download try %d failed (%s): %s' % (i + 1, mu, e))
            time.sleep(5)
    raise RuntimeError('download failed after %d tries across %d mirrors'
                       % (tries, len(mirrors)))


DATA = '/root/private_data/ag_news_train.csv'


def load_rows():
    if os.path.exists(DATA):
        rows = []
        with open(DATA, encoding='utf-8') as f:
            for r in csv.reader(f):
                if len(r) >= 3 and r[0].isdigit():
                    rows.append((int(r[0]) - 1, (r[1] + ' ' + r[2]).lower()))
        if len(rows) >= 100000:
            return rows
        log('existing file incomplete (%d rows), re-download' % len(rows))
        os.remove(DATA)
    log('downloading AG News (real external dataset, 120k real news)...')
    if os.path.exists(DATA + '.part'):
        os.remove(DATA + '.part')
    robust_download(
        'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras'
        '/master/data/ag_news_csv/train.csv', DATA)
    rows = []
    with open(DATA, encoding='utf-8') as f:
        for r in csv.reader(f):
            if len(r) >= 3 and r[0].isdigit():
                rows.append((int(r[0]) - 1, (r[1] + ' ' + r[2]).lower()))
    return rows


rows = load_rows()
log('ag_news rows=%d' % len(rows))
random.Random(7).shuffle(rows)
train = rows[:N_TRAIN]
dev = rows[N_TRAIN:N_TRAIN + DEV_N]
test = rows[N_TRAIN + DEV_N:N_TRAIN + DEV_N + TEST_N]

cnt = Counter(w for _, s in train for w in s.split()[:MAXLEN])
stoi = {'<pad>': 0, '<unk>': 1}
for w, c in cnt.most_common(VOCAB - 2):
    stoi[w] = len(stoi)


def encode(s):
    return [stoi.get(w, 1) for w in s.split()[:MAXLEN]]


def batchify(data, bs):
    for i in range(0, len(data) - bs + 1, bs):
        chunk = data[i:i + bs]
        xl = [encode(s) for _, s in chunk]
        L = max(len(x) for x in xl)
        x = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, xx in enumerate(xl):
            x[j, :len(xx)] = torch.tensor(xx)
        y = torch.tensor([l for l, _ in chunk])
        yield x.to(DEV), y.to(DEV)


# ---------------- 异构模块（零件，人类只提供这些） ----------------
class AttnBlk(nn.Module):
    """多头自注意力（无位置编码，靠词序无关的集合信息）"""

    def __init__(s):
        super().__init__()
        s.qkv = nn.Linear(D, D * 3)
        s.proj = nn.Linear(D, D)
        s.ln = nn.LayerNorm(D)
        s.h = 4

    def forward(s, x):
        B, T, _ = x.shape
        q, k, v = s.qkv(s.ln(x)).chunk(3, dim=-1)
        q = q.view(B, T, s.h, -1).transpose(1, 2)
        k = k.view(B, T, s.h, -1).transpose(1, 2)
        v = v.view(B, T, s.h, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1)
        return x + s.proj(o)


class GRUBlk(nn.Module):
    """双向 GRU（时序结构）"""

    def __init__(s):
        super().__init__()
        s.gru = nn.GRU(D, D // 2, batch_first=True, bidirectional=True)
        s.ln = nn.LayerNorm(D)

    def forward(s, x):
        o, _ = s.gru(s.ln(x))
        return x + o


class MLPBlk(nn.Module):
    """逐位置前馈（非线性容量）"""

    def __init__(s):
        super().__init__()
        s.ln = nn.LayerNorm(D)
        s.f = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))

    def forward(s, x):
        return x + s.f(s.ln(x))


class GateBlk(nn.Module):
    """门控混合（选择性信息通路）"""

    def __init__(s):
        super().__init__()
        s.ln = nn.LayerNorm(D)
        s.g = nn.Linear(D, D)

    def forward(s, x):
        return x * torch.sigmoid(s.g(s.ln(x)) + 1.0) * 1.5


BLOCKS = {'attn': AttnBlk, 'gru': GRUBlk, 'mlp': MLPBlk, 'gate': GateBlk}


class EvoNet(nn.Module):
    """基因组 -> 网络。跳跃接线 = 结构搜索空间的核心。"""

    def __init__(s, genome):
        super().__init__()
        s.emb = nn.Embedding(VOCAB, D, padding_idx=0)
        s.genome = genome
        s.blocks = nn.ModuleList([BLOCKS[b['type']]()
                                  for b in genome['blocks']])
        s.alphas = nn.ParameterList()
        for b in genome['blocks']:
            if b.get('skip') is not None:
                s.alphas.append(nn.Parameter(torch.tensor(0.5)))
        s.cls = nn.Sequential(nn.LayerNorm(D * 2), nn.Linear(D * 2, 4))

    def forward(s, x):
        outs = [s.emb(x)]
        ai = 0
        for i, spec in enumerate(s.genome['blocks']):
            inp = outs[-1]
            sk = spec.get('skip')
            if sk is not None:
                inp = inp + s.alphas[ai] * outs[sk]
                ai += 1
            outs.append(s.blocks[i](inp))
        f = torch.cat([outs[-1].mean(1), outs[-1].amax(1)], dim=-1)
        return s.cls(f)


# ---------------- 基因组操作（架构进化算子） ----------------
def sanitize(g):
    for i, b in enumerate(g['blocks']):
        sk = b.get('skip')
        if sk is None:
            continue
        if not isinstance(sk, int) or sk < 0 or sk > i - 1:
            b['skip'] = max(0, min(sk, i - 1)) if isinstance(sk, int) else None
    return g


def rand_genome():
    n = random.randint(2, 5)
    bl = [{'type': random.choice(list(BLOCKS)), 'skip': None}
          for _ in range(n)]
    for i in range(1, n):
        if random.random() < 0.4:
            bl[i]['skip'] = random.randint(0, i - 1)
    return sanitize({'blocks': bl})


def mutate(g):
    g = json.loads(json.dumps(g))
    op = random.choice(['add', 'del', 'type', 'skip', 'noskip'])
    bl = g['blocks']
    if op == 'add' and len(bl) < 6:
        bl.insert(random.randint(0, len(bl)),
                  {'type': random.choice(list(BLOCKS)), 'skip': None})
    elif op == 'del' and len(bl) > 2:
        bl.pop(random.randrange(len(bl)))
    elif op == 'type' and bl:
        bl[random.randrange(len(bl))]['type'] = random.choice(list(BLOCKS))
    elif op == 'skip' and len(bl) >= 2:
        i = random.randrange(1, len(bl))
        bl[i]['skip'] = random.randint(0, i - 1)
    elif op == 'noskip' and bl:
        bl[random.randrange(len(bl))]['skip'] = None
    return sanitize(g)


def gstr(g):
    return '->'.join(b['type'] + ('(%d)' % b['skip']
                                  if b.get('skip') is not None else '')
                     for b in g['blocks'])


def train_eval(genome, seed):
    """一个架构候选：权重从零训练，等预算 STEPS 步。"""
    torch.manual_seed(seed)
    random.seed(seed)
    m = EvoNet(sanitize(json.loads(json.dumps(genome)))).to(DEV)
    nparams = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    it = 0
    m.train()
    while it < STEPS:
        random.shuffle(train)
        for x, y in batchify(train, BS):
            loss = F.cross_entropy(m(x), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            it += 1
            if it >= STEPS:
                break

    def acc_on(data):
        m.eval()
        c = n = 0
        with torch.no_grad():
            for x, y in batchify(data, BS):
                c += (m(x).argmax(-1) == y).sum().item()
                n += y.numel()
        return c / n

    da, ta = acc_on(dev), acc_on(test)
    del m, opt
    gc.collect()
    torch.cuda.empty_cache()
    return da, ta, nparams


# ---------------- 1) 固定基线：人类默认架构，等预算 ----------------
log('== baselines: fixed human-default architectures, equal budget ==')
BASELINES = {
    'mlp_x3': {'blocks': [{'type': 'mlp', 'skip': None}] * 3},
    'gru_x3': {'blocks': [{'type': 'gru', 'skip': None}] * 3},
    'attn_x3': {'blocks': [{'type': 'attn', 'skip': None}] * 3},
    'hybrid': {'blocks': [{'type': 'mlp', 'skip': None},
                          {'type': 'attn', 'skip': None},
                          {'type': 'mlp', 'skip': None}]},
}
base_res = {}
for name, g in BASELINES.items():
    da, ta, np_ = train_eval(g, SEED)
    base_res[name] = {'dev': round(da, 4), 'test': round(ta, 4),
                      'params': np_}
    log('  %-8s dev=%.4f test=%.4f params=%.2fM' % (name, da, ta, np_ / 1e6))
    RESULT['baselines'] = base_res
    RESULT['stage'] = 'baseline_' + name
    save()

# ---------------- 2) 架构进化：拓扑跨代演化 ----------------
log('== architecture evolution: %d generations x population %d =='
    % (GENS, POP))
pop = [rand_genome() for _ in range(POP)]
pop_hist = []
for gen in range(GENS):
    RESULT['stage'] = 'gen_%d' % (gen + 1)
    save()
    scored = []
    for g in pop:
        da, ta, np_ = train_eval(g, SEED + 100 + gen)
        scored.append({'genome': g, 'arch': gstr(g), 'dev': round(da, 4),
                       'test': round(ta, 4), 'params': np_})
        log('  gen%d  %-46s dev=%.4f test=%.4f params=%.2fM'
            % (gen + 1, gstr(g), da, ta, np_ / 1e6))
    scored.sort(key=lambda s: -s['dev'])
    pop_hist.append(scored)
    RESULT['evolution'] = pop_hist
    save()
    pop = [s['genome'] for s in scored[:KEEP]]
    while len(pop) < POP:
        pop.append(mutate(random.choice(pop)))
    log('  gen%d done, top: %s (dev=%.4f)'
        % (gen + 1, scored[0]['arch'], scored[0]['dev']))

# ---------------- 3) 门禁：进化架构 vs 全部固定架构 ----------------
allc = [c for gen in pop_hist for c in gen]
best = max(allc, key=lambda c: c['dev'])
best_base = max(base_res.items(), key=lambda kv: kv[1]['dev'])
gate = best['test'] > max(v['test'] for v in base_res.values())
RESULT.update({
    'stage': 'done',
    'best_evolved': best,
    'best_baseline': {'name': best_base[0], **best_base[1]},
    'e3_gate': bool(gate),
    'ts_end': time.strftime('%F %T')})
save()
log('BEST EVOLVED : %s dev=%.4f test=%.4f'
    % (best['arch'], best['dev'], best['test']))
log('BEST BASELINE: %s test=%.4f' % (best_base[0], best_base[1]['test']))
log('E3 ARCH GATE: %s' % ('PASS' if gate else 'FAIL'))

gc.collect()
torch.cuda.empty_cache()
log('gpu released')
