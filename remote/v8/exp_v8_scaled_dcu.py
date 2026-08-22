# -*- coding: utf-8 -*-
"""v8 发育式 Transformer 扩大预算实验 —— SCNet 海光 DCU 优化版

相对 exp_v8_ablation_dcu.py 的差异（实验逻辑不变，仅性能优化 + 扩预算）：
  1. 预分词缓存：corpus 每行只 BPE 编码一次（编码确定性 -> 训练轨迹完全等价，纯提速 2.3x）
  2. 新增 fixed_mid(128) 配置：检验 dev 长到的 ~114 神经元是否即最优固定容量
  3. 增量保存（沿用）：每组跑完即写盘
  4. 启动时打印显存余量（共享卡，需监控溢出风险）

配置对比：
  dev        : hid_init=64, cap=256, 发育开启（分裂/修剪/定型）
  fixed_small: hid_init=64, cap=64,  发育关闭
  fixed_mid  : hid_init=128,cap=128, 发育关闭
  fixed_large: hid_init=256,cap=256, 发育关闭
"""
import os, sys, json, time, random, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import torch

from bionic_llm_v8 import BionicLLMv8, DevelopmentEngine
from bionic_llm_v7 import BPETokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print(f"[device] {torch.cuda.get_device_name(0)}", flush=True)
    f, t = torch.cuda.mem_get_info()
    print(f"[mem] free {f/2**30:.2f} GB / total {t/2**30:.2f} GB", flush=True)

CORPUS = os.path.join(HERE, 'corpus_zh_en.txt')

# ---- 预分词缓存 + device 搬运（合并补丁，核心代码零改动） ----
_orig_seq_for = DevelopmentEngine._seq_for
_tok_cache = {}

def _seq_for_cached(self, texts, n):
    lines = self.rng.sample(texts, min(n, len(texts)))
    max_len = self.block_size - 1
    rows = []
    for line in lines:
        ids = _tok_cache.get(line)
        if ids is None:
            ids = self.tok.encode(line)
            _tok_cache[line] = ids
        rows.append(self.tok.pad_to(ids[:max_len], max_len))
    X = torch.tensor([[self.tok.bos_id] + r for r in rows], dtype=torch.long)
    Y = torch.tensor([r + [self.tok.eos_id] for r in rows], dtype=torch.long)
    return X.to(DEVICE), Y.to(DEVICE)

DevelopmentEngine._seq_for = _seq_for_cached

# ---------------- 测试集（沿用 diag_teacher 24 题） ----------------
IN_ARITH = [
    ("3+4=", "7"), ("7+9=", "16"), ("5+5=", "10"), ("12-7=", "5"),
    ("20-9=", "11"), ("8*6=", "48"), ("9*9=", "81"), ("12*12=", "144"),
    ("13+27=", "40"), ("99-57=", "42"), ("15-9=", "6"), ("11*11=", "121"),
]
OUT_ARITH = [
    ("123+45=", "168"), ("999-123=", "876"), ("17*23=", "391"),
    ("45+67=", "112"), ("500-256=", "244"), ("13*47=", "611"),
]
ZH_FACT = [("中国的首都是", "北京"), ("太阳系有几大行星", "八大行星"),
           ("水的化学式是", "H2O")]
EN_FACT = [("The capital of France is", "Paris"),
           ("Water freezes at zero degrees", "Celsius"),
           ("The Earth orbits around", "the Sun")]
TEST = ([(p, a, "in") for p, a in IN_ARITH] +
        [(p, a, "out") for p, a in OUT_ARITH] +
        [(p, a, "zh") for p, a in ZH_FACT] +
        [(p, a, "en") for p, a in EN_FACT])


def load_corpus(seed):
    tok = BPETokenizer()
    with open(CORPUS, encoding='utf-8') as f:
        texts = [ln.strip() for ln in f if ln.strip()]
    tok.fit(texts)
    rng = random.Random(seed)
    rng.shuffle(texts)
    split = int(len(texts) * 0.85)
    return tok, texts[:split], texts[split:]


def make_model(tok, cfg, seed):
    kw = dict(n_layer=4, n_embd=128, block_size=128, n_head=4, head_dim=32,
              dropout=0.1, seed=seed)
    if cfg == 'dev':
        m = BionicLLMv8(tok, hid_cap=256, hid_init=64, **kw)
    elif cfg == 'fixed_small':
        m = BionicLLMv8(tok, hid_cap=64, hid_init=64, **kw)
    elif cfg == 'fixed_mid':
        m = BionicLLMv8(tok, hid_cap=128, hid_init=128, **kw)
    elif cfg == 'fixed_large':
        m = BionicLLMv8(tok, hid_cap=256, hid_init=256, **kw)
    else:
        raise ValueError(cfg)
    return m.to(DEVICE)


def teacher_forced_eval(model, tok):
    model.eval()
    stats = {'in': [0, 0, 0], 'out': [0, 0, 0], 'zh': [0, 0, 0],
             'en': [0, 0, 0]}
    with torch.no_grad():
        for p, a, cat in TEST:
            ids = tok.encode(p)
            x = torch.tensor([[tok.bos_id] + ids], dtype=torch.long,
                             device=DEVICE)
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            ans_tok = a[0]
            if ans_tok not in tok.stoi:
                continue
            aid = tok.stoi[ans_tok]
            rank = int((probs > probs[aid]).sum().item()) + 1
            s = stats[cat]
            s[0] += 1
            s[1] += rank == 1
            s[2] += rank <= 5
    t1 = sum(s[1] for s in stats.values())
    t5 = sum(s[2] for s in stats.values())
    return {'top1': t1, 'top5': t5, 'total': len(TEST),
            'per_cat': {k: {'n': v[0], 't1': v[1], 't5': v[2]}
                        for k, v in stats.items()}}


def argmax_generate(model, tok, prompt, max_new_tokens=8):
    model.eval()
    ids = tok.encode(prompt)[:model.block_size - 1]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([[tok.bos_id] + ids[-(model.block_size - 1):]],
                             dtype=torch.long, device=DEVICE)
            logits, _ = model(x)
            sel = int(logits[0, -1, :].argmax().item())
            if sel == tok.eos_id:
                break
            ids.append(sel)
    return tok.decode(ids)


def em(ans, out):
    a, o = ans.lower(), out.lower().strip()
    return o.startswith(a) or a in o.split()[:6]


def argmax_eval(model, tok):
    res = {"in": 0, "out": 0, "zh": 0, "en": 0, "total": 0}
    for p, a, cat in TEST:
        out = argmax_generate(model, tok, p, 8)
        ok = em(a, out)
        res[cat] += ok
        res["total"] += ok
    return res


def run_one(cfg, seed, n_steps, verbose=False):
    torch.manual_seed(seed)
    tok, tr, va = load_corpus(seed)
    model = make_model(tok, cfg, seed)
    eng = DevelopmentEngine(model, tok, tr, va, lr=1e-3, block_size=128,
                            n_batch=16, warmup_steps=40)
    if cfg != 'dev':
        eng.enable_dev = False
    t0 = time.time()
    hist = eng.train_loop(n_steps, verbose=verbose)
    wall = time.time() - t0

    tf = teacher_forced_eval(model, tok)
    gen = argmax_eval(model, tok)
    return {
        'cfg': cfg, 'seed': seed, 'n_steps': n_steps, 'wall_s': round(wall, 1),
        'device': DEVICE,
        'final_loss': hist['loss'][-1], 'best_val': eng._best_loss,
        'val_series': [float(x) for x in hist['val']],
        'loss_series': [float(x) for x in hist['loss']],
        'tf': tf, 'gen': gen,
        'active_neurons': model.n_active_neurons(),
        'n_split': eng.dev.n_split_total, 'n_prune': eng.dev.n_prune_total,
        'final_phase': eng.dev.phase,
        'active_series': [list(map(int, x)) for x in hist['active']],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+',
                    default=['dev', 'fixed_small', 'fixed_mid', 'fixed_large'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--steps', type=int, default=3200)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'v8_scaled.json'))
    args = ap.parse_args()
    if args.quick:
        args.steps = min(args.steps, 80)

    results = []
    if os.path.exists(args.out):        # 断点续跑：跳过已完成组
        with open(args.out, encoding='utf-8') as f:
            results = json.load(f)
        done = {(r['cfg'], r['seed'], r['n_steps']) for r in results}
    else:
        done = set()

    for cfg in args.configs:
        for sd in args.seeds:
            if (cfg, sd, args.steps) in done:
                print(f"skip {cfg}/{sd} (done)", flush=True)
                continue
            print(f"\n===== [{cfg}] seed={sd} steps={args.steps} =====",
                  flush=True)
            res = run_one(cfg, sd, args.steps, verbose=True)
            results.append(res)
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
            print(f"  -> wall={res['wall_s']}s tf_top1={res['tf']['top1']}/24 "
                  f"tf_top5={res['tf']['top5']}/24 gen={res['gen']['total']}/24 "
                  f"neurons={res['active_neurons']} "
                  f"split={res['n_split']} prune={res['n_prune']}", flush=True)

    print(f"\n[保存] {args.out}")


if __name__ == '__main__':
    main()
