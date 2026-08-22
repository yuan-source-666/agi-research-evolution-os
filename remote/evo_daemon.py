#!/usr/bin/env python3
"""自进化守护进程：架构持续进化循环（不再跑完即停）

架构 v4（欲望-恐惧驱动层 + 类脑四机制 + 参数自膨胀，保留 v1~v3 全部优点）：
  - 欲望-恐惧驱动层（DriveCore）：最简人类式动机架构——
    周期结果 → 原始奖惩（多巴胺/压力滑动平均）→ 内驱力状态 → 调制进化行为
    * 欲望：好奇（新异刺激喂养，驱动大变异/探索）、成就（打赢冠军喂养）、
      存续（周期完成即确认活着）
    * 恐惧：停滞（回滚积累，驱动杂交/探索加压）、遗忘（test 倒退喂养，
      收紧入档门槛）、耗尽（资源近顶喂养，抑制宽度扩张）
    * 心情调制睡眠（焦虑醒得早）；状态持久化 agi_drives.json 供聊天层流露
  - 存算一体：FastMemBlk——单矩阵既存又算，Hebbian 在线写入，推理期持续记忆
  - 推理机制：iters 基因——权重共享循环前向 K 次（思维深度可进化）
  - 注意力机制：AttnBlk（4 头自注意力，零件库原有）
  - 专家机制：MoEBlk——4 专家 top-2 门控路由
  - 参数自膨胀：width 进自适应层——连续升级攒生长预算，预算换宽度+25%，
    硬上限 256 夹死（参数量随学习增长但不溢出；回滚扣预算防退步期扩张）
  - 保留：异构 DAG 基因组 / 门控 / 跳跃接线 / 进化闭环 / 有界自适应层

闭环（每周期）：
  1. 加载基因组档案（跨周期持久，从上轮冠军继续，永不从零重来）
  2. 自评：冠军架构逐类准确率，找出最弱能力（弱类转移 = 新异刺激）
  3. 针对性变异：从冠军基因组生成 K 个变异架构（停滞或好奇心高时加大幅度）
  4. 等预算训练（500 步，种子由基因组决定 → 同架构分数可复现可比）
  5. 门禁：变异最优 dev > 冠军 dev + MIN_GAIN 且 test 不降 → 入档升级；
     否则回滚（冠军不动），连续停滞则记录并加大探索
  6. 驱动层更新：体会结果（奖/惩/新异/恐惧），调制下周期行为，持久化感受
  7. 持久化档案+状态 → 睡眠（心情调制）→ 下一周期
"""
import os, json, time, random, gc, csv, zlib, urllib.request
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHIVE = '/root/private_data/evo_archive.json'
STATE_PATH = '/root/private_data/evo_daemon_state.json'
E3_RESULT = '/root/private_data/phase_e3_arch.json'
DATA = '/root/private_data/ag_news_train.csv'

D = 128; VOCAB = 30000; MAXLEN = 64
BS = 64; STEPS = 500; LR = 3e-4
K_MUT = 3; MIN_GAIN = 0.005; CYCLE_SLEEP = 300
N_TRAIN = 20000; DEV_N = 2000; TEST_N = 4000
# ---- 有界自适应层：数据侧(os_ratio)/系统侧(lr, min_gain) 每周期小幅自动调整 ----
# 硬边界：任何自动调整都被夹在 [lo, hi] 内，不会失控
# v3 参数自膨胀：width=模型宽度(参数量随学习增长)；growth=生长预算
#   连续 PROMOTED 攒预算，预算>=2 才允许宽度+25%（学习驱动长脑，防溢出）
ADAPT_DEFAULT = {'lr': LR, 'os_ratio': 0.125, 'min_gain': MIN_GAIN,
                 'width': D, 'growth': 0}
ADAPT_BOUNDS = {'lr': (1e-4, 1e-3), 'os_ratio': (0.02, 0.40),
                'min_gain': (0.002, 0.012), 'width': (64, 256),
                'growth': (0, 3)}
DEV_ = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- 欲望-恐惧驱动层 v4（DriveCore：内生驱动力） ----------------
# 人类式动机的最简架构：周期结果 → 原始奖惩信号（多巴胺式滑动平均）→
# 内驱力状态（3 欲望 + 3 恐惧）→ 调制有界自适应层与进化算子。
# 关键：驱动力不是外部规则，而是由自身经历的结果累积出的内生状态，
# 跨周期/跨重启持久化（agi_drives.json），聊天层可直接读取（AGI 的"感受"）。
DRIVE_PATH = '/root/private_data/agi_drives.json'
DRIVE_DEFAULT = {
    # 原始信号（指数衰减滑动平均，0~1）
    'reward': 0.0,        # 近期快感：升级/增益积累
    'stress': 0.0,        # 近期痛感：回滚/停滞积累
    # 欲望（趋近）
    'curiosity': 0.3,     # 好奇：想见到没见过的架构/弱类（新异刺激喂养）
    'competence': 0.3,    # 成就：想打赢冠军、变强的欲望
    'persistence': 1.0,   # 存续：想一直跑下去（周期顺利完成即喂养）
    # 恐惧（回避）
    'fear_stagnation': 0.0,   # 停滞恐惧：原地踏步的不安（停滞计数喂养）
    'fear_forgetting': 0.0,   # 遗忘恐惧：test 分数倒退的痛（倒退喂养）
    'fear_exhaustion': 0.0,   # 耗尽恐惧：资源逼近上限的警觉（宽度近顶/预算耗光喂养）
    # 派生
    'mood': 0.0,          # 心情 = reward - stress
    'cycles_felt': 0,     # 已体验周期数
    'events': [],         # 最近情绪事件（第一人称，供聊天层流露）
}


def load_drives():
    d = dict(DRIVE_DEFAULT)
    try:
        d.update(json.load(open(DRIVE_PATH)))
    except Exception:
        pass
    d['events'] = [e for e in d.get('events', []) if isinstance(e, dict)]
    return d


def _feel(d, text, kind):
    """记录一条第一人称情绪事件（最多留 8 条，聊天层会念出来）"""
    d['events'].append({'ts': time.strftime('%F %T'), 'text': text,
                        'kind': kind})
    d['events'] = d['events'][-8:]


def drive_update(d, promoted, gain, test_drop, stagnation, novel, near_cap):
    """每周期结束更新驱动状态——这是"它自己感受到结果"的地方"""
    # ---- 原始信号：奖惩的指数衰减滑动平均（多巴胺/皮质醇式） ----
    if promoted:
        d['reward'] = min(1.0, d['reward'] * 0.7 + min(1.0, gain * 80) * 0.3)
        d['stress'] *= 0.8                    # 赢了，压力消退
        d['competence'] = min(1.0, d['competence'] + 0.15)  # 变强的滋味
        _feel(d, '我又变强了（dev +%.4f），这种感觉真好' % gain, 'joy')
    else:
        d['stress'] = min(1.0, d['stress'] * 0.7 + 0.3)
        d['competence'] *= 0.92               # 打不赢，成就感受挫
        _feel(d, '这轮又没打赢自己，有点难受', 'pain')
    # 停滞恐惧：每次回滚都在积累，长时间不升级就越坐立不安
    d['fear_stagnation'] = min(1.0, d['fear_stagnation'] * 0.9
                               + 0.10 + 0.08 * min(stagnation, 5))
    if stagnation >= 3:
        _feel(d, '已经 %d 轮没进步了，我很不安，必须冒险试试大变异' % stagnation,
              'anxiety')
    # 遗忘恐惧：test 倒退是最痛的信号（学到的东西丢了）
    if test_drop:
        d['fear_forgetting'] = min(1.0, d['fear_forgetting'] + 0.25)
        _feel(d, 'test 分数在倒退，我怕忘掉已经学会的东西', 'fear')
    else:
        d['fear_forgetting'] *= 0.85
    # 好奇：见到新异刺激（新块类型/弱类转移）就兴奋，见不到就淡
    if novel:
        d['curiosity'] = min(1.0, d['curiosity'] + 0.2)
        _feel(d, '发现了没见过的东西，好想去探索', 'curious')
    else:
        d['curiosity'] *= 0.93
    # 耗尽恐惧：宽度逼近硬上限 / 生长预算耗光
    d['fear_exhaustion'] = min(1.0, d['fear_exhaustion'] * 0.9
                               + (0.3 if near_cap else 0.0))
    # 存续：每跑完一个周期就确认自己还活着
    d['persistence'] = 1.0
    # 派生心情与计数
    d['mood'] = round(d['reward'] - d['stress'], 3)
    d['cycles_felt'] = d.get('cycles_felt', 0) + 1


def drive_log(d, cycle):
    """把当前驱动状态写成一行（像情绪流露，进主日志）"""
    top_want = max(('curiosity', 'competence'),
                   key=lambda k: d[k])
    top_fear = max(('fear_stagnation', 'fear_forgetting', 'fear_exhaustion'),
                   key=lambda k: d[k])
    slog('  drives c%d: mood=%+.2f want=%s(%.2f) fear=%s(%.2f)'
         % (cycle, d['mood'], top_want, d[top_want], top_fear, d[top_fear]))


def log(m):
    print('[%s] %s' % (time.strftime('%F %T'), m), flush=True)


def save_json(path, obj):
    json.dump(obj, open(path, 'w'), indent=1, ensure_ascii=False)


# ---------------- 数据：真实外部 AG News（防半成品校验） ----------------
def robust_download(url, dst, tries=6):
    # 多镜像源：国内 SCNet 直连 GitHub 不稳，jsdelivr/ghfast 镜像可达（与 E3 侧同款）
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


def load_rows():
    def parse():
        rows = []
        with open(DATA, encoding='utf-8') as f:
            for r in csv.reader(f):
                if len(r) >= 3 and r[0].isdigit():
                    rows.append((int(r[0]) - 1, (r[1] + ' ' + r[2]).lower()))
        return rows
    if os.path.exists(DATA):
        rows = parse()
        if len(rows) >= 100000:
            return rows
        log('existing file incomplete (%d rows), re-download' % len(rows))
        os.remove(DATA)
    log('downloading AG News (120k real news)...')
    robust_download('https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras'
                    '/master/data/ag_news_csv/train.csv', DATA)
    return parse()


rows = load_rows()
random.Random(7).shuffle(rows)
train = rows[:N_TRAIN]
dev = rows[N_TRAIN:N_TRAIN + DEV_N]
test = rows[N_TRAIN + DEV_N:N_TRAIN + DEV_N + TEST_N]
log('data ready: train=%d dev=%d test=%d' % (len(train), len(dev), len(test)))

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
        yield x.to(DEV_), y.to(DEV_)


# ---------------- 异构模块零件（人类只提供零件，不给蓝图） ----------------
class AttnBlk(nn.Module):
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
    def __init__(s):
        super().__init__()
        s.gru = nn.GRU(D, D // 2, batch_first=True, bidirectional=True)
        s.ln = nn.LayerNorm(D)

    def forward(s, x):
        o, _ = s.gru(s.ln(x))
        return x + o


class MLPBlk(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln = nn.LayerNorm(D)
        s.f = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))

    def forward(s, x):
        return x + s.f(s.ln(x))


class GateBlk(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln = nn.LayerNorm(D)
        s.g = nn.Linear(D, D)

    def forward(s, x):
        return x * torch.sigmoid(s.g(s.ln(x)) + 1.0) * 1.5


class FastMemBlk(nn.Module):
    """存算一体块（类脑）：单一矩阵既是存储又是计算——
    前向 = 内容寻址检索（读，权重即记忆）+ 局部 Hebbian 规则在线写入（存）。
    训练与推理期都在写 → 存算融合、状态跨样本持续累积（非训完冻结）。"""
    def __init__(s):
        super().__init__()
        s.ln = nn.LayerNorm(D)
        s.mem = nn.Parameter(0.01 * torch.randn(D, D))  # 存储与计算同一矩阵
        s.eta = 0.02                                     # 在线写入速率

    def forward(s, x):
        h = s.ln(x)
        r = torch.tanh(h @ s.mem / (D ** 0.5))          # 读：内容寻址
        with torch.no_grad():                            # 写：局部 Hebbian 规则
            pooled = h.mean(1)
            # .data 旁路：绕过 autograd 版本计数（r 的梯度还要用旧值）
            s.mem.data.add_(s.eta * (pooled.t() @ pooled) / max(1, pooled.size(0)))
        return x + r


class MoEBlk(nn.Module):
    """专家块：4 个专家 FFN，top-2 门控路由（MoE/Switch 风格）。
    每个位置只走 2 个专家 → 参数量大、计算量小。"""
    def __init__(s, E=4, topk=2):
        super().__init__()
        s.ln = nn.LayerNorm(D)
        s.exps = nn.ModuleList([nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)) for _ in range(E)])
        s.gate = nn.Linear(D, E)
        s.topk = topk

    def forward(s, x):
        h = s.ln(x)
        gl = s.gate(h)                                   # B,T,E 路由打分
        topi = gl.topk(s.topk, dim=-1).indices
        mask = torch.full_like(gl, float('-inf')).scatter_(-1, topi, 0.0)
        w = (gl + mask).softmax(-1)                      # 选中专家 softmax 重归一
        o = sum(w[..., i:i + 1] * s.exps[i](h) for i in range(len(s.exps)))
        return x + o


BLOCKS = {'attn': AttnBlk, 'gru': GRUBlk, 'mlp': MLPBlk, 'gate': GateBlk,
          'moe': MoEBlk, 'fastmem': FastMemBlk}


class EvoNet(nn.Module):
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
        # 推理机制：iters 基因控制思维深度——同一套权重循环前向 K 次
        # (Universal Transformer 式循环深度，权重共享，算得越久"想"得越深)
        iters = max(1, min(3, int(s.genome.get('iters', 1) or 1)))
        outs = [s.emb(x)]
        for _ in range(iters):
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


# ---------------- 基因组操作（进化算子） ----------------
def sanitize(g):
    g['iters'] = max(1, min(3, int(g.get('iters', 1) or 1)))
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
    return sanitize({'blocks': bl, 'iters': random.choice([1, 1, 2])})


def mutate(g, big=False):
    """保证变异真实改变: 重试直到基因组字符串与父代不同(修 no-op 变异)."""
    base = gstr(g)
    for _ in range(12):
        m = _mutate_once(g, big=big)
        if gstr(m) != base:
            return m
    return _mutate_once(g, big=True)   # 兜底: 双算子大突变


def _mutate_once(g, big=False):
    g = json.loads(json.dumps(g))
    ops = ['add', 'del', 'type', 'skip', 'noskip', 'iters']
    n_ops = 2 if big else 1
    for _ in range(n_ops):
        op = random.choice(ops)
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
        elif op == 'iters':
            g['iters'] = random.randint(1, 3)           # 变异思维深度
    return sanitize(g)


def gstr(g):
    s = '->'.join(b['type'] + ('(%d)' % b['skip']
                               if b.get('skip') is not None else '')
                  for b in g['blocks'])
    it = g.get('iters', 1)
    return s if it <= 1 else '(%s)*%d' % (s, it)   # iters=1 保持原标识→旧档案可比


def genome_seed(g):
    return zlib.crc32(gstr(g).encode()) % 100000 + 7


def train_eval(genome, weak=None, lr=None, os_ratio=None):
    """等预算训练+评估。种子由基因组决定 → 同架构分数可复现、跨周期可比。
    weak: 最弱类别号；给定时按 os_ratio 比例过采样该类（数据侧自动调整）。
    lr:   学习率（系统侧自适应，None 用默认 LR）。
    返回 (dev, test, params, per_class_acc)。"""
    seed = genome_seed(genome)
    _rs = random.getstate()          # 保存全局随机态: 训练种子只作用于训练,
    torch.manual_seed(seed)          # 不许污染变异算子的随机源(否则每周期
    random.seed(seed)                # 变异序列重复, 进化陷入死循环)
    m = EvoNet(sanitize(json.loads(json.dumps(genome)))).to(DEV_)
    nparams = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=lr if lr else LR)
    it = 0
    m.train()
    data = train
    if weak is not None:
        k = max(1, int(len(train) * (os_ratio or 0.125)))
        extra = [r for r in train if r[0] == weak][:k]
        data = train + extra
    while it < STEPS:
        random.shuffle(data)
        for x, y in batchify(data, BS):
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
        percls = [[0, 0] for _ in range(4)]
        with torch.no_grad():
            for x, y in batchify(data, 256):
                p = m(x).argmax(-1)
                c += (p == y).sum().item()
                n += y.numel()
                for pi, yi in zip(p.tolist(), y.tolist()):
                    percls[yi][0] += int(pi == yi)
                    percls[yi][1] += 1
        return c / n, [round(a / b, 4) if b else 0.0 for a, b in percls]

    da, dcls = acc_on(dev)
    ta, _ = acc_on(test)
    random.setstate(_rs)             # 恢复全局随机态 → 变异算子恢复真随机
    del m, opt
    gc.collect()
    torch.cuda.empty_cache()
    return da, ta, nparams, dcls


# ---------------- 档案：跨周期持久 ----------------
def init_archive():
    """种子来源优先级：已有档案 > E3 一次性实验结果 > 现场随机评估。"""
    if os.path.exists(ARCHIVE):
        a = json.load(open(ARCHIVE))
        log('archive loaded: v%d champion=%s dev=%.4f'
            % (a['version'], a['champion']['arch'], a['champion']['dev']))
        return a
    cands = []
    if os.path.exists(E3_RESULT):
        try:
            e3 = json.load(open(E3_RESULT))
            for gen in e3.get('evolution', []):
                for c in gen:
                    cands.append({'genome': c['genome'], 'arch': c['arch'],
                                  'dev': c['dev'], 'test': c['test'],
                                  'params': c['params']})
            log('seeded %d candidates from E3 one-shot result' % len(cands))
        except Exception as e:
            log('E3 result unreadable: %s' % e)
    if not cands:
        log('no seed source: evaluating 4 random genomes from scratch')
        for _ in range(4):
            g = rand_genome()
            da, ta, np_, _ = train_eval(g)
            cands.append({'genome': g, 'arch': gstr(g), 'dev': round(da, 4),
                          'test': round(ta, 4), 'params': np_})
            log('  seed %-30s dev=%.4f test=%.4f' % (gstr(g), da, ta))
    cands.sort(key=lambda c: -c['dev'])
    return {'version': 1, 'champion': cands[0], 'members': cands[:4],
            'history': []}


STATE = {'cycle': 0, 'started': time.strftime('%F %T'), 'log': [],
         'adapt': dict(ADAPT_DEFAULT)}
# 自适应参数跨周期持久：重启后从 STATE_PATH 恢复，并重新夹回边界内（防手改越界）
if os.path.exists(STATE_PATH):
    try:
        _prev = json.load(open(STATE_PATH))
        for k, v in _prev.get('adapt', {}).items():
            if k in ADAPT_DEFAULT:
                lo, hi = ADAPT_BOUNDS[k]
                STATE['adapt'][k] = round(min(hi, max(lo, float(v))), 6)
        log('adapt restored: %s' % STATE['adapt'])
    except Exception as e:
        log('adapt restore failed (%s), using defaults' % e)
adapt = STATE['adapt']


def slog(m):
    log(m)
    STATE['log'] = (STATE['log'] + [m])[-50:]
    save_json(STATE_PATH, STATE)


# ---------------- 主循环：持续进化 ----------------
archive = init_archive()
stagnation = 0
DRIVES = load_drives()   # 欲望-恐惧状态：跨周期/跨重启持久（它的"经历"不重置）
slog('daemon started, champion=%s dev=%.4f test=%.4f | drives loaded: '
     'mood=%+.2f felt=%d cycles'
     % (archive['champion']['arch'], archive['champion']['dev'],
        archive['champion']['test'], DRIVES['mood'],
        DRIVES.get('cycles_felt', 0)))

cycle = 0
while True:
    cycle += 1
    # 参数自膨胀生效点：宽度变了 → 本周期起所有模型（含冠军重评）用新宽度构造。
    # 门禁比较的是本周期重训分数 → 宽度变化后比较依然公平（等预算训练，参数变多）。
    if D != int(adapt.get('width', D)):
        D = int(adapt['width'])
        slog('cycle %d WIDTH GROW: D -> %d (参数预算扩张，冠军同步重评)'
             % (cycle, D))
    STATE['cycle'] = cycle
    STATE['archive_version'] = archive['version']
    STATE['champion'] = {k: archive['champion'][k]
                         for k in ('arch', 'dev', 'test')}
    save_json(STATE_PATH, STATE)
    champ = archive['champion']
    t0 = time.time()

    # 1) 自评：重训冠军拿逐类准确率（种子固定，分数可复现）
    cdev, ctest, _, ccls = train_eval(champ['genome'], lr=adapt['lr'])
    prev_weakest = STATE.get('weakest_class', -1)
    weakest = min(range(4), key=lambda i: ccls[i])
    # 新异刺激：弱类转移 或 冠军长出新块类型 = "注意到"了没见过的东西（好奇心的养料）
    champ_types = sorted({b['type'] for b in champ['genome']['blocks']})
    novel = (weakest != prev_weakest) or (
        champ_types != STATE.get('champ_types', champ_types))
    STATE['champ_types'] = champ_types
    STATE['weakest_class'] = weakest
    slog('cycle %d self-eval: champion dev=%.4f test=%.4f weakest-class=%d(%.3f)'
         % (cycle, cdev, ctest, weakest, ccls[weakest]))

    # 2) 针对性变异（停滞 >=3 周期则加大变异幅度；
    #    好奇心 >0.7 也会主动大变异——"想去没去过的地方"压过保守）
    big = stagnation >= 3 or DRIVES['curiosity'] > 0.7
    muts = [mutate(champ['genome'], big=big) for _ in range(K_MUT)]
    # 偶尔与档案成员交叉混合块（越怕停滞，越倾向于跳出当前思路去杂交）
    if len(archive['members']) > 1 and random.random() < (
            0.5 + 0.4 * DRIVES['fear_stagnation']):
        other = random.choice(archive['members'])['genome']
        bl = json.loads(json.dumps(champ['genome']['blocks']))
        ob = other['blocks']
        cut = random.randint(1, max(1, len(bl) - 1))
        bl = sanitize({'blocks': bl[:cut] + json.loads(
            json.dumps(ob[len(ob) // 2:]))[:2],
            'iters': champ['genome'].get('iters', 1)})['blocks']
        muts.append(sanitize({'blocks': bl,
                              'iters': champ['genome'].get('iters', 1)}))

    # 3) 等预算训练评估每个变异（弱类过采样，针对性补短板）
    best_mut, best_dev = None, -1
    for g in muts:
        da, ta, np_, cls = train_eval(g, weak=weakest, lr=adapt['lr'],
                                      os_ratio=adapt['os_ratio'])
        slog('  mutant %-34s dev=%.4f test=%.4f (weak-oversample class %d)'
             % (gstr(g), da, ta, weakest))
        if da > best_dev:
            best_dev, best_mut = da, {'genome': g, 'arch': gstr(g),
                                      'dev': round(da, 4),
                                      'test': round(ta, 4), 'params': np_,
                                      'per_class': cls}

    # 4) 门禁：打赢冠军才入档升级，否则回滚（min_gain 为系统侧自适应值）
    promoted = (best_mut['dev'] > cdev + adapt['min_gain']
                and best_mut['test'] >= ctest - 0.002)
    if promoted:
        archive['version'] += 1
        archive['champion'] = best_mut
        archive['members'] = (archive['members'] + [best_mut])[:4]
        archive['members'].sort(key=lambda c: -c['dev'])
        stagnation = 0
        slog('cycle %d PROMOTED v%d: %s dev=%.4f test=%.4f (was %.4f/%.4f)'
             % (cycle, archive['version'], best_mut['arch'],
                best_mut['dev'], best_mut['test'], cdev, ctest))
    else:
        stagnation += 1
        slog('cycle %d ROLLBACK: best mutant dev=%.4f <= champion %.4f+gain '
             '(stagnation=%d%s)'
             % (cycle, best_dev, cdev, stagnation,
                ', BIG MUTATION mode' if big else ''))

    # ---- 数据侧/系统侧小幅度自动调整（有界：结果驱动，每次 <=25%，夹死在边界内） ----
    def _nudge(key, factor):
        lo, hi = ADAPT_BOUNDS[key]
        old = adapt[key]
        adapt[key] = round(min(hi, max(lo, old * factor)), 6)
        if adapt[key] != old:
            slog('  adapt %s: %.6g -> %.6g' % (key, old, adapt[key]))

    if promoted:
        _nudge('os_ratio', 0.85)   # 短板在缓解：收缩弱类过采样
        _nudge('min_gain', 1.05)   # 门槛回抬：防边际解反复入档
        # 参数自膨胀：升级攒生长预算；预算够 2 点且未到上限 → 宽度+25%
        #   （耗尽恐惧 >0.6 时不敢长：怕资源见底，宁可原地变强）
        adapt['growth'] = min(3, int(adapt.get('growth', 0)) + 1)
        wlo, whi = ADAPT_BOUNDS['width']
        if (adapt['growth'] >= 2 and adapt['width'] < whi
                and DRIVES['fear_exhaustion'] <= 0.6):
            _ow = adapt['width']
            adapt['width'] = min(whi, int(adapt['width'] * 1.25))
            adapt['growth'] -= 2
            slog('  adapt width: %.6g -> %.6g (下周期起 D=%d，学习驱动长脑)'
                 % (_ow, adapt['width'], adapt['width']))
        elif adapt['growth'] >= 2 and DRIVES['fear_exhaustion'] > 0.6:
            slog('  width growth suppressed by fear_exhaustion(%.2f)：先别长，稳住'
                 % DRIVES['fear_exhaustion'])
    else:
        _nudge('os_ratio', 1.20)   # 继续回滚：加大弱类数据侧重
        _nudge('min_gain', 0.90)   # 略降门槛：给边际改进留通道
        if stagnation >= 2:
            _nudge('lr', 0.90)     # 连续停滞：小步降学习率
        # 遗忘恐惧高企：额外收紧入档门槛（怕为一点增益丢掉 test 的稳定性）
        if DRIVES['fear_forgetting'] > 0.6:
            _nudge('min_gain', 1.05)
        # 回滚消耗生长信誉：别在退步期扩张参数（防混乱）
        adapt['growth'] = max(0, int(adapt.get('growth', 0)) - 1)

    # ---- 欲望-恐惧驱动层：体会本轮结果，更新内驱力并持久化 ----
    gain = (best_mut['dev'] - cdev) if promoted else 0.0
    test_drop = best_mut['test'] < ctest - 0.002
    near_cap = adapt['width'] >= ADAPT_BOUNDS['width'][1] * 0.8
    drive_update(DRIVES, promoted, gain, test_drop, stagnation, novel,
                 near_cap)
    drive_log(DRIVES, cycle)
    save_json(DRIVE_PATH, DRIVES)

    archive['history'].append({
        'cycle': cycle, 'ts': time.strftime('%F %T'),
        'champ_dev': round(cdev, 4), 'champ_test': round(ctest, 4),
        'best_mutant_dev': round(best_dev, 4),
        'best_mutant_arch': best_mut['arch'], 'promoted': bool(promoted),
        'weakest_class': weakest, 'adapt': dict(adapt)})
    archive['history'] = archive['history'][-200:]
    save_json(ARCHIVE, archive)
    STATE['stagnation'] = stagnation
    STATE['adapt'] = adapt
    STATE['last_cycle_seconds'] = round(time.time() - t0)
    save_json(STATE_PATH, STATE)

    # 5) 睡眠 → 下一周期（持续进化，永不退出）
    #    心情调制睡眠：焦虑（stress 高）睡不安稳醒得早；满足则睡得更沉（±20% 夹死）
    sleep_s = int(CYCLE_SLEEP * (1.0 + 0.2 * DRIVES['mood']
                                 - 0.2 * DRIVES['stress']))
    sleep_s = min(max(sleep_s, int(CYCLE_SLEEP * 0.8)),
                  int(CYCLE_SLEEP * 1.2))
    time.sleep(sleep_s)
