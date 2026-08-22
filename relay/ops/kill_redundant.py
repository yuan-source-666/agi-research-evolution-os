# -*- coding: utf-8 -*-
"""杀掉多余进程：多余 E3(与 daemon 抢卡) + E1 wrapper。保留 evo_daemon。"""
import os, json, requests, websocket, uuid

BASE = 'https://REDACTED_JUPYTER_BASE'
TOKEN = 'REDACTED_JUPYTER_TOKEN'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)

CODE = (
    "import subprocess\n"
    "r = subprocess.run(['bash', '-lc',\n"
    "    'kill -9 16040 16041 2>/dev/null; sleep 2; "
    "ps aux | grep -E \"phase_e|evo_daemon\" | grep -v grep'],\n"
    "    capture_output=True, text=True)\n"
    "print(r.stdout)\nprint(r.stderr)\n"
)

r = requests.post(BASE + '/api/kernels', params={'token': TOKEN},
                  json={'name': 'python3'}, timeout=30)
r.raise_for_status()
kid = r.json()['id']
ws = websocket.create_connection(
    BASE.replace('https://', 'wss://', 1) + '/api/kernels/%s/channels?token=%s' % (kid, TOKEN),
    timeout=120)
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
    except Exception as e:
        out.append('[WS] ' + str(e))
        break
    msg = json.loads(raw)
    ch, mt = msg.get('channel'), msg.get('msg_type')
    if ch == 'iopub' and mt == 'stream':
        out.append(msg['content'].get('text', ''))
    if ch == 'iopub' and mt == 'error':
        out.append('ERR ' + str(msg['content'].get('ename')) + ':' + str(msg['content'].get('evalue')))
    if ch == 'shell' and mt == 'execute_reply':
        break
ws.close()
try:
    requests.delete(BASE + '/api/kernels/' + kid, params={'token': TOKEN}, timeout=15)
except Exception:
    pass
print(''.join(out))
