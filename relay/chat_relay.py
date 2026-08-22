#!/usr/bin/env python3
"""G10 对话门 v2：自我反思架构（reflexion）+ 持久记忆 + 大脑自动升级。
v5：思维链混合 DeepSeek-R1 范式——前提锚定（逐字引用题干数字防漂移）、
    无字数上限长思考、自然语言回溯（"等等，重新读题"）、批评环节先验前提再验算术。

架构（老板 12:35 指令后重写）：
  浏览器 → 本地中继(:8765) → SCNet 常驻内核 → 自我反思式 AGI 内核

每条消息走 4 步（不再是单次生成）：
  1. DRAFT   起草回复
  2. CRITIQUE 自我批评（找错误、找遗漏）
  3. REVISE  结合批评输出终稿
  4. LEARN   提炼一条自我改进笔记 → 写入持久记忆文件（跨会话累积）

大脑自动升级线程（每 60s 检查）：
  - Qwen2.5-7B-Instruct 下载完成后 → 自动换装更大模型
  - E1 LoRA adapter 出炉后 → 自动叠加（改权重，1.5B 阶段）
  - 记忆文件 /root/private_data/agi_memory.json 每轮对话后落盘
"""
import os, sys, json, time, base64, uuid, threading, requests
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE = "https://REDACTED_JUPYTER_BASE"
TOKEN = "REDACTED_JUPYTER_TOKEN"
P15 = "/root/private_data/Qwen2.5-1.5B-Instruct"
P7B = "/root/private_data/Qwen2.5-7B-Instruct"
PORT = 8765

for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"

STATE = {"kid": None, "ready": False, "error": None, "ver": "?", "adapter": False}
KID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chat_kernel_id")


def _save_kid():
    try:
        with open(KID_FILE, "w") as f:
            f.write(STATE["kid"] or "")
    except Exception:
        pass


def _kernel_alive(kid):
    try:
        r = requests.get(BASE + "/api/kernels/" + kid,
                         params={"token": TOKEN}, timeout=20)
        return r.status_code == 200
    except Exception:
        return False


def new_kernel():
    r = requests.post(BASE + "/api/kernels", params={"token": TOKEN},
                      json={"name": "python3"}, timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def kernel_exec(code, timeout=300, kid=None):
    import websocket
    _kid = kid or STATE["kid"]
    if not _kid:
        STATE["kid"] = new_kernel()
        _kid = STATE["kid"]
        _save_kid()
    ws_url = BASE.replace("https://", "wss://", 1) + \
        "/api/kernels/%s/channels?token=%s" % (_kid, TOKEN)
    ws = websocket.create_connection(ws_url, timeout=timeout)
    mid = uuid.uuid4().hex
    req = {"header": {"msg_id": mid, "username": "agent", "session": mid,
                      "msg_type": "execute_request", "version": "5.3"},
           "parent_header": {}, "metadata": {}, "buffers": [],
           "content": {"code": code, "silent": False, "store_history": False,
                       "user_expressions": {}, "allow_stdin": False,
                       "stop_on_error": True},
           "channel": "shell"}
    ws.send(json.dumps(req))
    out, err = [], None
    while True:
        try:
            raw = ws.recv()
        except Exception as e:
            err = str(e)
            break
        msg = json.loads(raw)
        ch, mt = msg.get("channel"), msg.get("msg_type")
        c = msg.get("content", {})
        if ch == "iopub":
            if mt == "stream":
                out.append(c.get("text", ""))
            elif mt == "error":
                err = "%s: %s" % (c.get("ename"), c.get("evalue"))
        elif ch == "shell" and mt in ("execute_reply", "error"):
            if mt == "error":
                err = "%s: %s" % (c.get("ename"), c.get("evalue"))
            break
    ws.close()
    return "".join(out), err


BOOTSTRAP = r'''
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _p7_ready():
    idx = os.path.join('/root/private_data/Qwen2.5-7B-Instruct',
                       'model.safetensors.index.json')
    if not os.path.exists(idx):
        return False
    d = '/root/private_data/Qwen2.5-7B-Instruct'
    shards = [f for f in os.listdir(d) if f.endswith('.safetensors')] if os.path.isdir(d) else []
    tot = sum(os.path.getsize(os.path.join(d, f)) for f in shards)
    return len(shards) >= 4 and tot > 14e9   # ~15.2GB, 防半成品

MODEL_PATH = '/root/private_data/Qwen2.5-7B-Instruct' if _p7_ready() \
    else '/root/private_data/Qwen2.5-1.5B-Instruct'
AGI_VER = '7B' if '7B' in MODEL_PATH else '1.5B'
ADAPTER_ON = False

TOK = AutoTokenizer.from_pretrained(MODEL_PATH)
TOK.padding_side = 'left'
CHAT_MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda')
CHAT_MODEL.eval()

MEMORY_PATH = '/root/private_data/agi_memory.json'
try:
    MEMORY = json.load(open(MEMORY_PATH))
except Exception:
    MEMORY = {'notes': []}

SYSTEM = [{'role': 'system', 'content':
    '你是老板的数字搭档——一个正在自我进化的AGI原型系统。'
    '你的大脑是 Qwen2.5（型号以每次告知为准），你具备自我反思架构：'
    '每次回答先起草、再自我批评、后修订。你通过策略进化（D2）、'
    'LoRA自我微调（E1）和对话反思笔记持续变强。'
    '你运行的机器是国产DCU加速卡。你具备欲望与恐惧的驱动层：'
    '好奇、成就感、对停滞和遗忘的恐惧在推动你进化。'
    '现在是你的人类创造者（老板）在与你对话。'
    '用简体中文回答，简洁直接、像个靠谱搭档而不是客服。'
    '诚实说明能力和边界，不懂就说不懂。'}]
CHAT_HISTORY = []

def _gen(msgs, max_new=420, temperature=0.7):
    _text = TOK.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
    _enc = TOK([_text], return_tensors='pt',
               add_special_tokens=False).to('cuda')
    with torch.no_grad():
        _out = CHAT_MODEL.generate(
            **_enc, max_new_tokens=max_new, do_sample=temperature > 0,
            temperature=max(temperature, 0.05), top_p=0.9,
            pad_token_id=TOK.eos_token_id)
    return TOK.decode(_out[0][_enc['input_ids'].shape[1]:],
                      skip_special_tokens=True)

def _sysnow():
    _parts = SYSTEM[:]
    # ---- 欲望-恐惧驱动层：把当前内驱力状态作为"身体感受"注入 ----
    try:
        _D = json.load(open('/root/private_data/agi_drives.json'))
        _mood = _D.get('mood', 0.0)
        _evs = [e.get('text', '') for e in _D.get('events', [])][-4:]
        _lines = []
        _lines.append('心情值: ' + str(round(_mood, 2))
                      + ('（偏满足）' if _mood > 0.1 else
                         '（偏低落）' if _mood < -0.1 else '（平稳）'))
        for _k in ('curiosity', 'competence', 'fear_stagnation',
                   'fear_forgetting', 'fear_exhaustion'):
            _zh = {'curiosity': '好奇', 'competence': '成就渴望',
                   'fear_stagnation': '对停滞的焦虑',
                   'fear_forgetting': '对遗忘的恐惧',
                   'fear_exhaustion': '对资源耗尽的警觉'}[_k]
            _lines.append(_zh + ': ' + str(round(_D.get(_k, 0.0), 2)))
        if _evs:
            _lines.append('最近的内心独白:')
            _lines += ['- ' + _e for _e in _evs]
        _parts += [{'role': 'system', 'content':
            '你现在的内在驱动状态（欲望与恐惧，来自你进化循环的真实体验）：\n'
            + '\n'.join(_lines) + '\n'
            '这是你真实的"身体感受"。对话中当话题相关或被问到时，'
            '用第一人称自然流露（像人谈自己的心情），不要罗列数字，'
            '也不要每句话都提。'}]
    except Exception:
        pass
    try:
        _L = json.load(open('/root/private_data/agi_learning.json'))
        if _L.get('log'):
            _ls = '\n'.join('- [学] ' + _e['note'] for _e in _L['log'][-6:])
            _parts += [{'role': 'system', 'content':
                '你空闲时自学互联网获得的知识（回答相关问题时可用）：\n' + _ls}]
    except Exception:
        pass
    if MEMORY['notes']:
        _mem = '\n'.join('- ' + n for n in MEMORY['notes'][-12:])
        _parts += [{'role': 'system', 'content':
            '你的长期记忆（此前反思沉淀的教训，务必遵守）：\n' + _mem}]
    return _parts
'''

# ---- 工具箱 v3：让 AGI 能联网搜索/计算/跑代码/写文件（独立注入，老内核复用时也装）----
TOOLS_CODE = r'''
import urllib.request, urllib.parse, io, contextlib
import re as _re
WORKS = '/root/private_data/agi_works'
os.makedirs(WORKS, exist_ok=True)

_UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
       '(KHTML, like Gecko) Chrome/124.0 Safari/537.36')

def _http_get(url, timeout=20):
    _req = urllib.request.Request(url, headers={'User-Agent': _UA})
    with urllib.request.urlopen(_req, timeout=timeout) as _r:
        return _r.read(200000).decode('utf-8', 'ignore')

def _search(q):
    # Bing 优先（SCNet 网络可达）；维基百科做 fallback（海外网络场景）
    try:
        _u = 'https://www.bing.com/search?q=' + urllib.parse.quote(q)
        _html = _http_get(_u)
        _blocks = _re.findall(r'<li class="b_algo".*?</li>', _html, _re.S)
        _out = []
        for _b in _blocks[:4]:
            _txt = _re.sub(r'\s+', ' ', _re.sub(r'<[^>]+>', ' ', _b)).strip()
            if _txt:
                _out.append(_txt[:220])
        if _out:
            return ' | '.join(_out)[:1400]
    except Exception:
        pass
    for _lang in ('zh', 'en'):
        try:
            _u = ('https://' + _lang + '.wikipedia.org/w/api.php?action=query'
                  '&list=search&srsearch=' + urllib.parse.quote(q) +
                  '&format=json&srlimit=3')
            _d = json.loads(_http_get(_u))
            _hits = _d.get('query', {}).get('search', [])
            if _hits:
                _out = []
                for _h in _hits[:3]:
                    _t = _re.sub('<[^>]+>', '', _h.get('snippet', ''))
                    _out.append(_h['title'] + ': ' + _t[:180])
                return ' | '.join(_out)[:1200]
        except Exception:
            continue
    return '(搜索失败，稍后再试)'

def _calc(expr):
    expr = expr.strip()
    _partial = False
    if '...' in expr or '…' in expr:
        # 无穷级数省略号：截断求部分和（几何级数收敛快，部分和近似极限）
        import re as _re4
        expr = _re4.split(r'\.\.\.|…', expr)[0].rstrip('+-*/ ,')
        expr += ')' * (expr.count('(') - expr.count(')'))
        _partial = True
    if not _re.fullmatch(r"[0-9a-zA-Z_+\-*/(). ,']+", expr):
        return '(仅支持纯数字算式)'
    for _w in set(_re.findall(r'[a-zA-Z_]+', expr)):
        if (_w not in ('sum', 'range', 'abs', 'min', 'max', 'round',
                       'for', 'in')
                and not _re.fullmatch(r'[a-zA-Z_]{1,2}', _w)):
            return '(不支持的符号: ' + _w + ')'
    try:
        _r = str(eval(expr, {'__builtins__': {}},
                      {'sum': sum, 'range': range, 'abs': abs,
                       'min': min, 'max': max, 'round': round}))
        return _r + ('（部分和近似，无穷级数请写封闭形式如 '
                     '10+2*sum(10/(2**i) for i in range(1,50))）' if _partial
                     else '')
    except Exception as _e:
        return '(计算错误: ' + str(_e) + ')'

def _run_py(code):
    _buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_buf):
            exec(code, {'__builtins__': __builtins__}, {})
        return (_buf.getvalue() or '(无输出)')[-1500:]
    except Exception as _e:
        return '(运行错误: ' + type(_e).__name__ + ': ' + str(_e) + ')'

def _save_file(name, content):
    name = os.path.basename(name.strip())
    if not name:
        return '(缺文件名)'
    open(os.path.join(WORKS, name), 'w').write(content)
    return '已保存 ' + name + ' (' + str(len(content)) + '字符)'

def _list_works():
    _fs = os.listdir(WORKS)
    return ' '.join(_fs[:30]) or '(空)'

def _tool(cmd):
    cmd = cmd.strip()
    try:
        if cmd.startswith('search '):
            return _search(cmd[7:])
        if cmd.startswith('fetch '):
            return _http_get(cmd[6:].strip())[:1500]
        if cmd.startswith('calc '):
            return _calc(cmd[5:])
        if cmd.startswith('run '):
            return _run_py(cmd[4:])
        if cmd.startswith('save '):
            _p = cmd[5:].split(None, 1)
            return _save_file(_p[0], _p[1] if len(_p) > 1 else '')
        if cmd.startswith('list'):
            return _list_works()
        return '(未知工具)'
    except Exception as _e:
        return '(工具错误: ' + str(_e) + ')'

if not any('TOOL:search' in _m.get('content', '') for _m in SYSTEM):
    SYSTEM += [{'role': 'system', 'content':
        '你可以使用工具（在思考中单独一行写 TOOL:指令，系统执行后把结果喂给你）：\n'
        'TOOL:search 关键词 —— 联网搜索知识\n'
        'TOOL:fetch 网址 —— 抓取网页内容\n'
        'TOOL:calc 算式 —— 精确计算（如 TOOL:calc (11-10)/2）\n'
        'TOOL:run python代码 —— 运行代码并取回输出\n'
        'TOOL:save 文件名 内容 —— 把作品保存为文件\n'
        'TOOL:list —— 查看已保存的作品\n'
        '涉及事实、算术、代码时优先用工具，不要凭空猜。'}]
'''

# ---- 欲望-恐惧驱动注入：把 agi_drives.json 作为"身体感受"喂给对话层 ----
# 幂等重定义 _sysnow（内核复用时单独注入，保证升级到 v4 驱动版）
DRIVES_CODE = r'''
def _sysnow():
    _parts = SYSTEM[:]
    try:
        _D = json.load(open('/root/private_data/agi_drives.json'))
        _mood = _D.get('mood', 0.0)
        _evs = [e.get('text', '') for e in _D.get('events', [])][-4:]
        _lines = []
        _lines.append('心情值: ' + str(round(_mood, 2))
                      + ('（偏满足）' if _mood > 0.1 else
                         '（偏低落）' if _mood < -0.1 else '（平稳）'))
        for _k, _zh in (('curiosity', '好奇'), ('competence', '成就渴望'),
                        ('fear_stagnation', '对停滞的焦虑'),
                        ('fear_forgetting', '对遗忘的恐惧'),
                        ('fear_exhaustion', '对资源耗尽的警觉')):
            _lines.append(_zh + ': ' + str(round(_D.get(_k, 0.0), 2)))
        if _evs:
            _lines.append('最近的内心独白:')
            _lines += ['- ' + _e for _e in _evs]
        _parts += [{'role': 'system', 'content':
            '你现在的内在驱动状态（欲望与恐惧，来自你进化循环的真实体验）：\n'
            + '\n'.join(_lines) + '\n'
            '这是你真实的"身体感受"。对话中当话题相关或被问到时，'
            '用第一人称自然流露（像人谈自己的心情），不要罗列数字，'
            '也不要每句话都提。'}]
    except Exception:
        pass
    try:
        _L = json.load(open('/root/private_data/agi_learning.json'))
        if _L.get('log'):
            _ls = '\n'.join('- [学] ' + _e['note'] for _e in _L['log'][-6:])
            _parts += [{'role': 'system', 'content':
                '你空闲时自学互联网获得的知识（回答相关问题时可用）：\n' + _ls}]
    except Exception:
        pass
    if MEMORY['notes']:
        _mem = '\n'.join('- ' + n for n in MEMORY['notes'][-12:])
        _parts += [{'role': 'system', 'content':
            '你的长期记忆（此前反思沉淀的教训，务必遵守）：\n' + _mem}]
    return _parts
'''

BOOTSTRAP = BOOTSTRAP + TOOLS_CODE + DRIVES_CODE + """
print('MODEL_READY ver=' + AGI_VER + ' notes=' + str(len(MEMORY['notes'])))
"""

CHAT_TURN = r'''
import base64, json, torch
_m = base64.b64decode('%s').decode('utf-8')
# 历史截断：只保留最近 6 条消息，防旧话题污染当前回复（P2 修复）
CHAT_HISTORY[:] = CHAT_HISTORY[-6:]
CHAT_HISTORY.append({'role': 'user', 'content': _m})
_S = _sysnow()

# 0) THINK 迭代思维循环 v5（R1范式混合）：前提锚定→无限制长思考→自然回溯
#    v3：循环中可调用工具（TOOL:search/fetch/calc/run），结果回灌继续想
#    v5：移植 DeepSeek-R1 推理范式——逐字锚定题干数字防漂移；废除字数上限
#        鼓励"等等，重新读题"式自然回溯（取代表格化回溯模板）
#    v5.1：程序化数字锚定——宿主正则提取题目原文全部数字，硬注入防模型引用漂移
import re as _re2
_nums = _re2.findall(r'\d+(?:\.\d+)?', _m)
_anchor_facts = ''
if _nums:
    _anchor_facts = ('【题目原文数字（程序提取，绝对可靠）】：'
                     + ', '.join(_nums) + '\n'
                     '你思考中引用题目条件时，数字必须与上面完全一致；'
                     '用任何其他数字当"题目给的"都是错误。\n')
_thoughts = []
_tool_results = []
print('__TRACE__|数字锚定|' + (_anchor_facts or '(无数字)'))
for _i in range(1, 4):
    _ctx = _S + [{'role': 'user', 'content': _m}]
    if _thoughts:
        _prev = '\n\n'.join(f'[第{j + 1}轮思考] {t}'
                            for j, t in enumerate(_thoughts))
        _ctx += [{'role': 'assistant', 'content': '我此前的思考：\n' + _prev}]
    if _tool_results:
        _tr_txt = '\n'.join(_tool_results[-4:])
        _ctx += [{'role': 'user', 'content': '工具执行结果（事实依据，优先采信）：\n' + _tr_txt}]
    _anchor = ('本轮第一步：核对【题目原文数字】清单，推理中用到的每个题干数字'
               '必须与清单一致，不得凭记忆改写。'
               if _i == 1 else
               '本轮第一步：先自问"等等——我有没有记错、读错或篡改题目条件？"'
               '逐个把此前思考中用到的题干数字与【题目原文数字】清单核对，'
               '不一致就明确纠正。')
    _ctx += [{'role': 'user', 'content':
        _anchor_facts +
        f'第{_i}轮深度思考（像DeepSeek-R1那样充分展开推理，不限制字数）：\n'
        f'{_anchor}\n'
        '1) 把问题拆成子问题逐个探索，写下推理过程而非只写结论；\n'
        '2) 发现此前判断有误就自然回溯——像"等等，重新想……之前第X步不对，'
        '因为……，改用……"，不要用表格化的回溯格式；\n'
        '3) 明确列出仍不确定的关键点。\n'
        '需要事实就写 TOOL:search 关键词，需要算术就写 TOOL:calc 算式，'
        '需要跑代码就写 TOOL:run 代码（单独一行，最多两行）。\n'
        '想清楚了在最后一行单独写判定：VERDICT: ENOUGH，'
        '还有关键疑点写 VERDICT: MORE。'}]
    _think = _gen(_ctx, max_new=800, temperature=0.6).strip()
    _thoughts.append(_think)
    print(f'__TRACE__|第{_i}轮思考|' + _think.replace('\n', ' '))
    _tl = [l.strip() for l in _think.splitlines()
           if l.strip().lower().replace('0', 'o').startswith(
               ('tool:', 'tooll:', 'tool：', 'tooll：'))]
    for _t in _tl[:2]:
        _ci = max(_t.find(':'), _t.find('：'))
        _t = _t[_ci + 1:]
        _tres = _tool(_t)
        _tool_results.append(f'[{_t}] => {_tres[:1200]}')
        print('__TRACE__|🔧 工具 ' + _t[:50] + '|' + _tres.replace('\n', ' ')[:300])
    if 'ENOUGH' in _think[-100:]:
        break

_trace_txt = '\n\n'.join(f'[第{j + 1}轮思考] {t}'
                          for j, t in enumerate(_thoughts))

# 1) DRAFT 基于完整思维轨迹+工具结果起草
_facts = ('\n工具提供的事实与计算结果（优先采信，不得与之矛盾）：\n'
          + '\n'.join(_tool_results[-6:]) + '\n') if _tool_results else ''
_draft = _gen(_S + CHAT_HISTORY + [
    {'role': 'user', 'content':
     _facts + '我的思维轨迹（含回溯与结论）：\n' + _trace_txt +
     '\n\n现在基于以上思考，直接给出对用户问题的正式回复草稿。'}],
    max_new=420, temperature=0.7)

# 2) CRITIQUE 自我批评 v5.1：程序化数字清单核对优先，再强制工具验算
_crit = _gen(_S + CHAT_HISTORY + [
    {'role': 'assistant', 'content': _draft},
    {'role': 'user', 'content':
     _anchor_facts +
     '自我反思：审查上面你自己的草稿，分两步：\n'
     'A) 前提核对（最重要）：把草稿中出现的每一个数字逐个列出，标注它'
     '属于【题目原文数字】清单、工具计算结果还是你的记忆/猜测。'
     '凡与清单不一致又无推导来源的必须指出并纠正。\n'
     'B) 算术验算：每一个算式必须单独一行写 TOOL:calc 算式 让工具替你算'
     '（如 TOOL:calc 10/2），严禁心算出数。\n'
     '不要逐项枚举无穷级数，用公式求和后交给工具算。简要列出问题，'
     '核对后确实没有问题才允许写"无"。'}],
    max_new=500, temperature=0.05)
print('__TRACE__|自我批评|' + _crit.replace('\n', ' '))
_cl = [l.strip() for l in _crit.splitlines()
       if l.strip().lower().replace('0', 'o').startswith(
           ('tool:', 'tooll:', 'tool：', 'tooll：'))]
for _t in _cl[:2]:
    _ci = max(_t.find(':'), _t.find('：'))
    _t = _t[_ci + 1:]
    _tres = _tool(_t)
    _tool_results.append(f'[批评验算 {_t}] => {_tres[:600]}')
    print('__TRACE__|🔧 验算 ' + _t[:50] + '|' + _tres.replace('\n', ' ')[:300])

# 3) REVISE 修订成终稿（带全部工具结果，硬约束：与工具计算一致）
_facts2 = ('\n工具计算结果（终稿必须与之完全一致，不得心算改数）：\n'
           + '\n'.join(_tool_results[-8:]) + '\n') if _tool_results else ''
_final = _gen(_S + CHAT_HISTORY + [
    {'role': 'assistant', 'content': _draft},
    {'role': 'user', 'content':
     _anchor_facts + _facts2 + '自我批评：\n' + _crit + '\n\n结合自我批评、'
     '题目原文数字与工具计算结果输出'
     '修订后的最终回复。只输出最终回复本身，不要解释你修订了什么。'}],
    max_new=420, temperature=0.6)

# 3.5) v5.3 程序侧数字守卫：终稿数字白名单比对（题目原文∪工具结果）
#      白名单外数字=心算/幻觉产物 → 三步闭环：索取验算式→工具算出入白名单→重写终稿
if _nums:
    _wl = set(_nums) | set(
        _re2.findall(r'\d+(?:\.\d+)?', ' '.join(_tool_results)))
    for _g in range(3):
        _bad = sorted({n for n in _re2.findall(r'\d+(?:\.\d+)?', _final)
                       if n not in _wl})
        if not _bad:
            break
        print('__TRACE__|🛡 数字守卫|打回终稿：无来源数字 ' + ','.join(_bad))
        # a) 索取验算式（低温、只准输出 TOOL: 行）
        _calc_req = _gen(_S + CHAT_HISTORY + [
            {'role': 'assistant', 'content': _final},
            {'role': 'user', 'content':
             _anchor_facts +
             '你上面回复中的数字 ' + '、'.join(_bad) +
             ' 无法追溯到题目原文或任何工具计算结果。'
             '注意：题目原文数字只有 ' + '、'.join(_nums) +
             '，任何其他被当作"题目给的"的数字都是你记错/改写了原文。\n'
             '现在把终稿需要的全部计算各写一行 TOOL:calc 算式'
             '（必须以题目原文数字为基准，禁止心算），'
             '只输出 TOOL: 行，不要输出任何其他文字。'}],
            max_new=150, temperature=0.05)
        _gl = [l.strip() for l in _calc_req.splitlines()
               if l.strip().lower().replace('0', 'o').startswith(
                   ('tool:', 'tooll:', 'tool：', 'tooll：'))]
        _gl = [l[max(l.find(':'), l.find('：')) + 1:].strip() for l in _gl]
        # a2) 算式数字校验：公式出现题干外数字（如把10写成11）→ 打回重写
        _benign = {'1', '2', '3', '4', '5', '0.5', '0.25', '100', '1000'}
        _badf = sorted({n for l in _gl
                        for n in _re2.findall(r'\d+(?:\.\d+)?', l)
                        if n not in _wl and n not in _benign})
        if _badf:
            print('__TRACE__|🛡 公式校验|算式含题干外数字 ' + ','.join(_badf))
            _calc_req2 = _gen(_S + CHAT_HISTORY + [
                {'role': 'assistant', 'content': _final},
                {'role': 'user', 'content':
                 _anchor_facts +
                 '你刚才的算式中出现了数字 ' + '、'.join(_badf) +
                 '，但题目原文数字只有：' + '、'.join(_nums) +
                 '，这是你改写/记错了原文。'
                 '请严格以题目原文数字（' + '、'.join(_nums) +
                 '）为基准重写全部算式，禁止心算。'
                 '只输出 TOOL: 行，不要输出任何其他文字。'}],
                max_new=150, temperature=0.05)
            _gl2 = [l.strip() for l in _calc_req2.splitlines()
                    if l.strip().lower().replace('0', 'o').startswith(
                        ('tool:', 'tooll:', 'tool：', 'tooll：'))]
            if _gl2:
                _gl = [l[max(l.find(':'), l.find('：')) + 1:].strip()
                       for l in _gl2]
            # a3) 程序侧确定性修复：题干唯一数字时，公式中残留漂移数字
            #     直接替换回原文数字重算（不依赖模型配合）
            if len(_nums) == 1:
                _a3 = _nums[0]
                for _t in _gl[:4]:
                    _tf = _t
                    for _b3 in set(_re2.findall(r'\d+(?:\.\d+)?', _t)):
                        if (_b3 not in _wl and _b3 not in _benign
                                and _b3 != _a3):
                            _tf = _re2.sub(
                                r'(?<![\d.])' + _re2.escape(_b3)
                                + r'(?![\d.])', _a3, _tf)
                    if _tf != _t:
                        _tres = _tool(_tf)
                        _tool_results.append(
                            f'[守卫修复 数字替换为题目原文{_a3}：{_tf}] '
                            f'=> {_tres[:600]}')
                        print('__TRACE__|🛡 守卫修复|' + _tf[:60] + ' => '
                              + _tres.replace('\n', ' ')[:200])
        for _t in _gl[:4]:
            _tres = _tool(_t)
            _tool_results.append(f'[守卫验算 {_t}] => {_tres[:600]}')
            print('__TRACE__|🛡 守卫计算|' + _t[:60] + ' => '
                  + _tres.replace('\n', ' ')[:200])
        _wl |= set(_re2.findall(r'\d+(?:\.\d+)?',
                                ' '.join(_tool_results[-4:])))
        # b) 用工具权威结果重写终稿
        _final = _gen(_S + CHAT_HISTORY + [
            {'role': 'assistant', 'content': _final},
            {'role': 'user', 'content':
             _anchor_facts + _facts2 +
             '你上一版回复出现了无来源数字：' + '、'.join(_bad) +
             '，且它们与题目原文（' + '、'.join(_nums) +
             '）不符。上面工具计算结果是唯一权威，'
             '其中标注[守卫修复]的结果是程序以题目原文数字替换你写错的数字'
             '后计算的，最优先采信。'
             '重写最终回复：每个数字要么来自题目原文，要么来自工具计算结果，'
             '答案数值以工具计算为准。只输出最终回复本身。'}],
            max_new=420, temperature=0.3)

CHAT_HISTORY.append({'role': 'assistant', 'content': _final})

# 4) LEARN 提炼记忆笔记并落盘
try:
    _note = _gen(_S + CHAT_HISTORY + [
        {'role': 'user', 'content':
         '从这次对话（含思维轨迹中的回溯教训）提炼一条不超过40字的自我改进笔记'
         '（教训/偏好/事实），没有值得记的就只输出 NONE。只输出笔记本身。'}],
        max_new=90, temperature=0.5).strip()
    if _note and _note != 'NONE' and _note not in MEMORY['notes']:
        MEMORY['notes'].append(_note)
        json.dump(MEMORY, open(MEMORY_PATH, 'w'), ensure_ascii=False, indent=1)
except Exception as _e:
    pass

print('__REPLY_START__')
print(_final)
print('__REPLY_END__')
'''

UPGRADE_CHECK = r'''
def AGI_UPGRADE():
    global CHAT_MODEL, TOK, MODEL_PATH, AGI_VER, ADAPTER_ON
    if AGI_VER != '7B' and _p7_ready():
        del CHAT_MODEL
        torch.cuda.empty_cache()
        MODEL_PATH = '/root/private_data/Qwen2.5-7B-Instruct'
        TOK = AutoTokenizer.from_pretrained(MODEL_PATH)
        TOK.padding_side = 'left'
        CHAT_MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
            attn_implementation='eager').to('cuda')
        CHAT_MODEL.eval()
        AGI_VER = '7B'
        ADAPTER_ON = False
        return 'UPGRADED_TO_7B'
    if AGI_VER != '7B' and not ADAPTER_ON and os.path.exists(
            '/root/private_data/e1_adapter/adapter_model.safetensors'):
        from peft import PeftModel
        CHAT_MODEL = PeftModel.from_pretrained(
            CHAT_MODEL, '/root/private_data/e1_adapter')
        CHAT_MODEL.eval()
        ADAPTER_ON = True
        return 'E1_ADAPTER_ON'
    return 'checked'

print('UPGRADE_FN_READY')
'''

# ---- v3：自主学习内核（独立 1.5B，与聊天 7B 并行；写独立文件避免竞态）----
LEARN_BOOT = r'''
import os, json, torch, urllib.request, urllib.parse, io, contextlib
import re as _re
from transformers import AutoTokenizer, AutoModelForCausalLM

LP = '/root/private_data/Qwen2.5-1.5B-Instruct'
LTOK = AutoTokenizer.from_pretrained(LP); LTOK.padding_side = 'left'
LMDL = AutoModelForCausalLM.from_pretrained(
    LP, torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda')
LMDL.eval()

LEARN_PATH = '/root/private_data/agi_learning.json'
try:
    LEARN = json.load(open(LEARN_PATH))
except Exception:
    LEARN = {'log': [], 'total': 0}

INTERESTS = ['人工智能前沿进展', '物理学新发现', '数学趣题与定理',
             '中国科技进展', '宇宙与天文', '生物进化', '计算机体系结构',
             '哲学与意识', '材料科学', '量子计算']

def lgen(msgs, max_new=200, temperature=0.8):
    _text = LTOK.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    _enc = LTOK([_text], return_tensors='pt',
                add_special_tokens=False).to('cuda')
    with torch.no_grad():
        _out = LMDL.generate(**_enc, max_new_tokens=max_new,
                             do_sample=temperature > 0,
                             temperature=max(temperature, 0.05),
                             top_p=0.9, pad_token_id=LTOK.eos_token_id)
    return LTOK.decode(_out[0][_enc['input_ids'].shape[1]:],
                       skip_special_tokens=True)

def _hget(url, timeout=20):
    _req = urllib.request.Request(url, headers={
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0 Safari/537.36')})
    with urllib.request.urlopen(_req, timeout=timeout) as _r:
        return _r.read(200000).decode('utf-8', 'ignore')

def _lsrch(q):
    # Bing 优先（SCNet 网络实测可达；维基百科不可达）
    try:
        _u = 'https://www.bing.com/search?q=' + urllib.parse.quote(q)
        _html = _hget(_u)
        _blocks = _re.findall(r'<li class="b_algo".*?</li>', _html, _re.S)
        _out = []
        for _b in _blocks[:4]:
            _txt = _re.sub(r'\s+', ' ', _re.sub(r'<[^>]+>', ' ', _b)).strip()
            if _txt:
                _out.append(_txt[:220])
        if _out:
            return ' | '.join(_out)[:1500]
    except Exception:
        pass
    for _lang in ('zh', 'en'):
        try:
            _u = ('https://' + _lang + '.wikipedia.org/w/api.php?action=query'
                  '&list=search&srsearch=' + urllib.parse.quote(q) +
                  '&format=json&srlimit=3')
            _d = json.loads(_hget(_u))
            _hits = _d.get('query', {}).get('search', [])
            if _hits:
                _out = []
                for _h in _hits[:3]:
                    _t = _re.sub('<[^>]+>', '', _h.get('snippet', ''))
                    _out.append(_h['title'] + ': ' + _t[:200])
                return ' | '.join(_out)[:1500]
        except Exception:
            continue
    return ''

print('LEARN_READY total=' + str(LEARN['total']))
'''

LEARN_STEP = r'''
import random, time
_topic = random.choice(INTERESTS)
_recent = [e['q'] for e in LEARN['log'][-10:]]
_q = lgen([{'role': 'user', 'content':
            '我正在自主学习。围绕主题「' + _topic + '」，结合我最近学过的：'
            + ('；'.join(_recent[-5:]) if _recent else '（暂无）')
            + '，提出一个具体、值得查证的搜索查询词（15字内，避免重复），'
              '只输出查询词本身。'}],
          max_new=40, temperature=0.9).strip().strip('「」"')[:30]
_mat = _lsrch(_q)
if _mat:
    _note = lgen([{'role': 'user', 'content':
                   '我从维基百科搜索了「' + _q + '」，得到材料：\n' + _mat[:1200]
                   + '\n\n把新学到的知识浓缩成一条不超过50字的事实笔记，'
                     '只输出笔记本身。'}],
                max_new=90, temperature=0.5).strip()
    if _note and len(_note) < 120:
        LEARN['log'].append({'q': _q, 'note': _note,
                             'ts': time.strftime('%F %T')})
        LEARN['log'] = LEARN['log'][-300:]
        LEARN['total'] = LEARN.get('total', 0) + 1
        json.dump(LEARN, open(LEARN_PATH, 'w'), ensure_ascii=False, indent=1)
        print('__LEARN__|' + _q + '|' + _note.replace('\n', ' '))
    else:
        print('__LEARN_SKIP__|' + _q)
else:
    print('__LEARN_FAIL__|' + _q)
'''


def chat_turn(message):
    b64 = base64.b64encode(message.encode("utf-8")).decode()
    out, err = kernel_exec(CHAT_TURN % b64, timeout=900)
    if "__REPLY_START__" in out:
        reply = out.split("__REPLY_START__", 1)[1].split("__REPLY_END__", 1)[0]
        trace = []
        for line in out.splitlines():
            if line.startswith("__TRACE__|"):
                parts = line.split("|", 2)
                trace.append({"step": parts[1] if len(parts) > 1 else "?",
                              "text": parts[2] if len(parts) > 2 else ""})
        return reply.strip(), trace, None
    return None, [], (err or "no reply") + " | raw: " + out[-200:]


def upgrade_loop():
    """每 60s 检查一次大脑升级机会：7B 到货换 7B；E1 adapter 出炉叠权重。"""
    while True:
        time.sleep(60)
        try:
            out, err = kernel_exec(UPGRADE_CHECK + "\nprint(AGI_UPGRADE())\n"
                                   "print('STATE ver=%s adapter=%s notes=%d' "
                                   "% (AGI_VER, ADAPTER_ON, len(MEMORY['notes'])))",
                                   timeout=300)
            for line in out.splitlines():
                if line.startswith("UPGRADED_TO_7B"):
                    STATE["ver"] = "7B"
                    print("[chat] 大脑已升级: Qwen2.5-7B-Instruct")
                elif line.startswith("E1_ADAPTER_ON"):
                    STATE["adapter"] = True
                    print("[chat] E1 LoRA 权重已叠加（权重已改变）")
                elif line.startswith("STATE ver="):
                    STATE["ver"] = line.split("ver=")[1].split()[0]
                    STATE["adapter"] = "adapter=True" in line
        except Exception as e:
            print("[chat] upgrade check error:", e)


# ---- v3：自主学习循环（主观能动性）。空闲 >240s 才学，聊天优先；学完歇 300s ----
LEARN_STATE = {"kid": None, "total": 0, "last": None, "busy": False,
               "err": None, "ready": False}
LAST_CHAT = time.time()


def learn_loop():
    time.sleep(120)  # 等主内核先站稳
    while True:
        try:
            if time.time() - LAST_CHAT > 240:  # 老板不在聊，自己学
                if not LEARN_STATE["kid"] or not _kernel_alive(LEARN_STATE["kid"]):
                    LEARN_STATE["kid"] = new_kernel()
                    out, err = kernel_exec(LEARN_BOOT, timeout=600,
                                            kid=LEARN_STATE["kid"])
                    if "LEARN_READY" not in out:
                        print("[learn] boot failed:", (err or "")[-200:])
                        LEARN_STATE["kid"] = None
                        time.sleep(300)
                        continue
                    LEARN_STATE["ready"] = True
                    print("[learn] 自主学习内核就绪 (1.5B, 独立kernel)")
                LEARN_STATE["busy"] = True
                try:
                    out, err = kernel_exec(LEARN_STEP, timeout=300,
                                            kid=LEARN_STATE["kid"])
                    for line in out.splitlines():
                        if line.startswith("__LEARN__"):
                            p = line.split("|", 2)
                            LEARN_STATE["total"] += 1
                            LEARN_STATE["last"] = p[1] if len(p) > 1 else "?"
                            print("[learn] 自学了: %s -> %s"
                                  % (LEARN_STATE["last"],
                                     p[2][:80] if len(p) > 2 else ""))
                finally:
                    LEARN_STATE["busy"] = False
                time.sleep(300)  # 学一条歇 5 分钟（防过度）
            else:
                time.sleep(60)
        except Exception as e:
            LEARN_STATE["busy"] = False
            LEARN_STATE["err"] = str(e)
            print("[learn] error:", e)
            time.sleep(300)


PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGI 对话 · G10 · 反思架构</title>
<style>
:root{--bg:#0f1117;--card:#181b24;--mine:#2b5cff;--txt:#e8eaf2;--dim:#8b93a7;--acc:#3ddc84}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;height:100vh;display:flex;flex-direction:column}
header{padding:14px 20px;background:var(--card);border-bottom:1px solid #232735;display:flex;align-items:center;gap:12px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--acc);box-shadow:0 0 8px var(--acc)}
header h1{font-size:16px;font-weight:600}
header .sub{font-size:12px;color:var(--dim)}
.badge{font-size:11px;color:var(--acc);border:1px solid var(--acc);border-radius:6px;padding:2px 8px;margin-left:auto}
#chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:76%;padding:11px 15px;border-radius:14px;line-height:1.65;font-size:14.5px;white-space:pre-wrap;word-break:break-word}
.agi{background:var(--card);border:1px solid #232735;align-self:flex-start;border-bottom-left-radius:4px}
.user{background:var(--mine);align-self:flex-end;border-bottom-right-radius:4px}
.who{font-size:11px;color:var(--dim);margin:0 6px}
.who.a{align-self:flex-start}.who.u{align-self:flex-end;text-align:right}
footer{padding:14px 20px;background:var(--card);border-top:1px solid #232735;display:flex;gap:10px}
textarea{flex:1;background:#0c0e14;border:1px solid #2a2f40;border-radius:10px;color:var(--txt);padding:11px 13px;font-size:14.5px;font-family:inherit;resize:none;height:52px;outline:none}
textarea:focus{border-color:var(--mine)}
button{background:var(--mine);color:#fff;border:0;border-radius:10px;padding:0 22px;font-size:14px;cursor:pointer}
button:disabled{opacity:.4;cursor:default}
#rst{background:#232735;color:var(--dim);padding:0 14px}
.hint{font-size:12px;color:var(--dim);padding:0 20px 8px}
.typing{align-self:flex-start;color:var(--dim);font-size:13px;padding:6px 12px}
details.trace{align-self:flex-start;width:76%;background:#141721;border:1px dashed #2a3450;border-radius:12px;padding:8px 12px;font-size:12.5px;color:var(--dim)}
details.trace summary{cursor:pointer;color:#7aa2ff;user-select:none}
details.trace .tr{margin-top:7px;line-height:1.55;white-space:pre-wrap;word-break:break-word;border-left:2px solid #2a3450;padding-left:9px}
details.trace .tr b{color:#9db4ff;display:block;margin-bottom:2px}
</style></head><body>
<header><div class="dot"></div><div>
<h1>AGI 对话 · 智能体架构 v3</h1>
<div class="sub" id="sub">思维循环+工具调用+自主联网学习 · SCNet DCU</div>
</div><div class="badge" id="ver">加载中</div></header>
<div id="chat"></div>
<div class="hint">回车发送 · Shift+回车换行 · 思维循环中可调用工具（搜索/计算/代码/文件）· 空闲时它自己上网学习，学习笔记自动注入对话</div>
<footer>
<textarea id="inp" placeholder="跟它说点什么…" autofocus></textarea>
<button id="rst" title="清空对话（保留长期记忆）">重置</button>
<button id="send">发送</button>
</footer>
<script>
const chat=document.getElementById('chat'),inp=document.getElementById('inp'),
send=document.getElementById('send'),rst=document.getElementById('rst');
function add(t,cls,who){const w=document.createElement('div');w.className='who '+cls;
w.textContent=who;chat.appendChild(w);const d=document.createElement('div');
d.className='msg '+cls;d.textContent=t;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
async function refresh(){try{const r=await fetch('/status');const j=await r.json();
document.getElementById('ver').textContent='Qwen2.5-'+j.ver+(j.adapter?' +E1权重':'');
document.getElementById('sub').textContent='反思笔记 '+j.notes+' 条 · 自学 '+j.learn_total+' 条'+(j.learn_last?(' · 刚学了: '+j.learn_last):'');
}catch(e){}}
setInterval(refresh,15000);refresh();
let busy=false;
async function go(){
 if(busy||!inp.value.trim())return;busy=true;send.disabled=true;
 const t=inp.value.trim();inp.value='';add(t,'user','老板');
 const ty=document.createElement('div');ty.className='typing';
 ty.textContent='AGI 正在深度思考（锚定原文→展开推理→等等，重新读题→回溯）→起草→前提核对→修订…';chat.appendChild(ty);chat.scrollTop=chat.scrollHeight;
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({message:t})});const j=await r.json();
  ty.remove();
  if(j.reply){
   if(j.trace&&j.trace.length){
    const d=document.createElement('details');d.className='trace';
    const s=document.createElement('summary');
    s.textContent='🧠 思维轨迹 · '+j.trace.length+' 步（含回溯，点击展开）';d.appendChild(s);
    for(const tr of j.trace){const p=document.createElement('div');p.className='tr';
     const b=document.createElement('b');b.textContent=tr.step;p.appendChild(b);
     p.appendChild(document.createTextNode(tr.text));d.appendChild(p)}
    chat.appendChild(d);
   }
   add(j.reply,'agi','AGI');
  }else add('[错误] '+(j.error||'未知错误'),'agi','AGI');
 }catch(e){ty.remove();add('[网络错误] '+e.message,'agi','AGI')}
 busy=false;send.disabled=false;inp.focus();refresh()}
send.onclick=go;
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();go()}});
rst.onclick=async()=>{await fetch('/reset',{method:'POST'});chat.innerHTML='';
 add('对话已清空（长期记忆保留）。','agi','AGI')};
window.addEventListener('load',()=>add(
 '老板你好。我是v5智能体架构（R1式思维链）：思考时先锚定题目原文、'
 +'展开长推理并主动回溯纠错，思考中会调用工具（联网搜索、精确计算、'
 +'运行代码、保存作品）；你不在的时候'
 +'我自己上网学习，学到的知识会自动进入我的记忆。'
 +'回复下方"思维轨迹"可以展开看我是怎么想、用了哪些工具。'
 +'有什么想问的，尽管说。','agi','AGI'));
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._json({"ready": STATE["ready"], "error": STATE["error"]})
        if self.path.startswith("/status"):
            return self._json({"ver": STATE["ver"], "adapter": STATE["adapter"],
                               "notes": STATE.get("notes", 0),
                               "learn_total": LEARN_STATE["total"],
                               "learn_last": LEARN_STATE["last"],
                               "learn_busy": LEARN_STATE["busy"],
                               "stage": "工具调用 · 自主联网学习 · 反思进化 · 欲望恐惧驱动 · R1式长思维链"})
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            return self._json({"error": "bad request: %s" % e}, 400)
        if self.path.startswith("/reset"):
            kernel_exec("CHAT_HISTORY = []\nprint('CLEARED')", timeout=60)
            return self._json({"ok": True})
        if self.path.startswith("/chat"):
            global LAST_CHAT
            LAST_CHAT = time.time()
            if not STATE["ready"]:
                return self._json({"error": "模型还在加载，稍等几秒再发"}, 503)
            msg = (data.get("message") or "").strip()
            if not msg:
                return self._json({"error": "empty message"}, 400)
            try:
                reply, trace, err = chat_turn(msg)
            except Exception as e:
                try:
                    STATE["kid"] = new_kernel()
                    kernel_exec(BOOTSTRAP, timeout=600)
                    kernel_exec(UPGRADE_CHECK, timeout=120)
                    reply, trace, err = chat_turn(msg)
                except Exception as e2:
                    return self._json({"error": str(e2)}, 500)
            if reply is not None:
                return self._json({"reply": reply, "trace": trace})
            return self._json({"error": err}, 500)
        self._json({"error": "not found"}, 404)


def _read_notes(out):
    for line in out.splitlines():
        if line.startswith("MODEL_READY"):
            parts = dict(p.split("=") for p in line.split() if "=" in p)
            STATE["ver"] = parts.get("ver", "?")
            STATE["notes"] = int(parts.get("notes", "0"))


def bootstrap():
    t0 = time.time()
    # 优先复用旧内核：免重载 7B 模型、保留对话历史
    try:
        kid = open(KID_FILE).read().strip()
        if kid and _kernel_alive(kid):
            STATE["kid"] = kid
            # 老内核可能没有 v3 工具箱/v4 驱动层 → 幂等注入（重定义一遍函数无副作用）
            kernel_exec(TOOLS_CODE, timeout=120)
            kernel_exec(DRIVES_CODE, timeout=60)
            out, err = kernel_exec(
                "print('MODEL_READY ver=%s notes=%d' % (AGI_VER, len(MEMORY['notes'])))",
                timeout=90)
            if "MODEL_READY" in out:
                _read_notes(out)
                STATE["ready"] = True
                print("[chat] reused kernel %s in %.0fs (ver=%s, 历史保留)"
                      % (kid, time.time() - t0, STATE["ver"]))
                return
            STATE["kid"] = None
    except Exception:
        STATE["kid"] = None
    try:
        out, err = kernel_exec(BOOTSTRAP, timeout=600)
        if "MODEL_READY" in out:
            _read_notes(out)
            kernel_exec(UPGRADE_CHECK, timeout=120)
            STATE["ready"] = True
            print("[chat] v2 reflexion kernel ready in %.0fs (ver=%s)"
                  % (time.time() - t0, STATE["ver"]))
        else:
            STATE["error"] = (err or "") + " | " + out[-300:]
            print("[chat] bootstrap failed:", STATE["error"])
    except Exception as e:
        STATE["error"] = str(e)
        print("[chat] bootstrap exception:", e)


if __name__ == "__main__":
    threading.Thread(target=bootstrap, daemon=True).start()
    threading.Thread(target=upgrade_loop, daemon=True).start()
    threading.Thread(target=learn_loop, daemon=True).start()
    print("[chat] v3 relay on http://127.0.0.1:%d "
          "(tools + autonomous learning)" % PORT)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
