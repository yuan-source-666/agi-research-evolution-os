# -*- coding: utf-8 -*-
"""Launch background download of Qwen2.5-7B-Instruct on SCNet (hf-mirror)."""
import os, json, requests, websocket, uuid

BASE = 'https://REDACTED_JUPYTER_BASE'
TOKEN = 'REDACTED_JUPYTER_TOKEN'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)

r = requests.post(BASE + '/api/kernels', params={'token': TOKEN},
                  json={'name': 'python3'}, timeout=30)
r.raise_for_status()
kid = r.json()['id']
ws = websocket.create_connection(
    BASE.replace('https://', 'wss://', 1) + '/api/kernels/%s/channels?token=%s' % (kid, TOKEN),
    timeout=120)

CODE = r'''
import subprocess
# idempotent: skip if already downloaded
chk = subprocess.run(['bash','-lc',
  'test -f /root/private_data/Qwen2.5-7B-Instruct/model.safetensors.index.json && echo EXISTS || echo MISSING'],
  capture_output=True, text=True).stdout.strip()
print('7B status:', chk)
if chk == 'MISSING':
    cmd = ('export HF_ENDPOINT=https://hf-mirror.com; '
           'cd /root/private_data && '
           'python3 -c "from huggingface_hub import snapshot_download; '
           "snapshot_download('Qwen/Qwen2.5-7B-Instruct', "
           "local_dir='/root/private_data/Qwen2.5-7B-Instruct')\" "
           '> /root/private_data/dl_7b.log 2>&1 & echo started pid $!')
    out = subprocess.run(['bash','-lc', cmd], capture_output=True, text=True)
    print(out.stdout, out.stderr)
else:
    print('already present')
'''

mid = uuid.uuid4().hex
req = {'header': {'msg_id': mid, 'username': 'agent', 'session': mid,
                  'msg_type': 'execute_request', 'version': '5.3'},
       'parent_header': {}, 'metadata': {}, 'buffers': [],
       'content': {'code': CODE, 'silent': False, 'store_history': False,
                   'user_expressions': {}, 'allow_stdin': False,
                   'stop_on_error': True},
       'channel': 'shell'}
ws.send(json.dumps(req))
out = []
while True:
    try:
        raw = ws.recv()
    except Exception as e:
        out.append('[WS closed] ' + str(e)); break
    msg = json.loads(raw)
    ch, mt = msg.get('channel'), msg.get('msg_type')
    c = msg.get('content', {})
    if ch == 'iopub' and mt == 'stream':
        out.append(c.get('text', ''))
    if ch == 'iopub' and mt == 'error':
        out.append('ERR ' + str(c.get('ename')) + ':' + str(c.get('evalue')))
    if ch == 'shell' and mt in ('execute_reply', 'error'):
        break
ws.close()
requests.delete(BASE + '/api/kernels/' + kid, params={'token': TOKEN}, timeout=30)
print(''.join(out))
