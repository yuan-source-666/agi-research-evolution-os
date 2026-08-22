# -*- coding: utf-8 -*-
"""一次查清远端全局:进程(是否多实验抢卡) + E3 日志 + GPU + 守护进程状态"""
import os, json, requests, websocket, uuid

BASE = 'https://REDACTED_JUPYTER_BASE'
TOKEN = 'REDACTED_JUPYTER_TOKEN'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)

CODE = r'''
import subprocess, json
print('== PROC ==')
print(subprocess.run(['bash','-lc',
    'ps aux | grep -E "phase_e|evo_daemon|launch_|waiter" | grep -v grep'],
    capture_output=True, text=True).stdout)
print('== E3 LOG ==')
print(subprocess.run(['bash','-lc',
    'tail -12 /root/private_data/phase_e3_arch.log 2>/dev/null'],
    capture_output=True, text=True).stdout)
print('== GPU ==')
print(subprocess.run(['bash','-lc',
    'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'],
    capture_output=True, text=True).stdout)
print('== DAEMON STATE ==')
print(open('/root/private_data/evo_daemon_state.json').read()[:300]
      if __import__('os').path.exists('/root/private_data/evo_daemon_state.json') else '(none)')
print('== DATA FILE ==')
print(subprocess.run(['bash','-lc',
    'ls -la /root/private_data/ag_news_train.csv* 2>/dev/null; '
    'wc -l /root/private_data/ag_news_train.csv 2>/dev/null'],
    capture_output=True, text=True).stdout)
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
