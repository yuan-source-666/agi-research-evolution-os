# -*- coding: utf-8 -*-
"""拉取架构能力全量数据: E3 终局 JSON + 进化档案 + 守护进程状态"""
import os, json, requests, websocket, uuid

BASE = 'https://[REDACTED-ENDPOINT]:58043/jupyter-forward/[REDACTED-ID]'
TOKEN = '[REDACTED-TOKEN]'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)

CODE = r'''
import json, os

def show(path, keys=None, maxlen=1500):
    if not os.path.exists(path):
        print('-- %s: (missing)' % path); return None
    d = json.load(open(path))
    print('-- %s --' % path)
    if keys:
        for k in keys:
            print('  %s:' % k, json.dumps(d.get(k), ensure_ascii=False)[:maxlen])
    return d

e3 = show('/root/private_data/phase_e3_arch.json')
if e3:
    print('  E3 all keys:', list(e3.keys()))
    for k in ('baselines', 'gate', 'e3_gate', 'best'):
        if k in e3:
            print('  %s:' % k, json.dumps(e3[k], ensure_ascii=False)[:600])

arc = show('/root/private_data/evo_archive.json')
if arc:
    print('  archive keys:', list(arc.keys()))
    ch = arc.get('champion', {})
    print('  champion:', json.dumps({k: ch.get(k) for k in ('genome','dev','test','version') if k in ch}, ensure_ascii=False)[:400])
    print('  members:', len(arc.get('members', [])), ' history:', len(arc.get('history', [])))

st = show('/root/private_data/evo_daemon_state.json')
if st:
    for line in st.get('log', [])[-25:]:
        print('   |', line)
'''

r = requests.post(BASE + '/api/kernels', params={'token': TOKEN},
                  json={'name': 'python3'}, timeout=30)
r.raise_for_status()
kid = r.json()['id']
ws = websocket.create_connection(
    BASE.replace('https://', 'wss://', 1) + '/api/kernels/%s/channels?token=%s' % (kid, TOKEN),
    timeout=180)
mid = uuid.uuid4().hex
req = {'header': {'msg_id': mid, 'username': 'agent', 'session': mid,
                  'msg_type': 'execute_request', 'version': '5.3'},
       'parent_header': {}, 'metadata': {}, 'buffers': [],
       'content': {'code': CODE, 'silent': False, 'store_history': False,
                   'user_expressions': {}, 'allow_stdin': False, 'stop_on_error': True},
       'channel': 'shell'}
ws.send(json.dumps(req))
out = []
while True:
    try:
        raw = ws.recv()
    except Exception:
        break
    msg = json.loads(raw)
    if msg.get('channel') == 'iopub' and msg.get('msg_type') == 'stream':
        out.append(msg['content'].get('text', ''))
    if msg.get('channel') == 'iopub' and msg.get('msg_type') == 'error':
        out.append('ERR ' + str(msg['content'].get('ename')) + ':' + str(msg['content'].get('evalue')))
    if msg.get('channel') == 'shell' and msg.get('msg_type') == 'execute_reply':
        break
ws.close()
try:
    requests.delete(BASE + '/api/kernels/' + kid, params={'token': TOKEN}, timeout=15)
except Exception:
    pass
print(''.join(out))
