# -*- coding: utf-8 -*-
"""重部署 evo_daemon: contents API 上传修复版 -> 杀旧 -> nohup 启动"""
import os, json, base64, requests

BASE = 'https://[REDACTED-ENDPOINT]:58043/jupyter-forward/[REDACTED-ID]'
TOKEN = '[REDACTED-TOKEN]'
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = '*'


def api_put(path, content_b64):
    r = requests.put(BASE + '/api/contents' + path, params={'token': TOKEN},
                     json={'content': content_b64, 'format': 'base64',
                           'type': 'file'}, timeout=120)
    r.raise_for_status()
    return r.json()


def run_code(code, timeout=120):
    import uuid, websocket
    r = requests.post(BASE + '/api/kernels', params={'token': TOKEN},
                      json={'name': 'python3'}, timeout=30)
    r.raise_for_status()
    kid = r.json()['id']
    ws_url = BASE.replace('https://', 'wss://', 1) + \
        '/api/kernels/%s/channels?token=%s' % (kid, TOKEN)
    ws = websocket.create_connection(ws_url, timeout=timeout)
    msg_id = uuid.uuid4().hex
    req = {'header': {'msg_id': msg_id, 'username': 'agent', 'session': msg_id,
                      'msg_type': 'execute_request', 'version': '5.3'},
           'parent_header': {}, 'metadata': {}, 'buffers': [],
           'content': {'code': code, 'silent': False, 'store_history': False,
                       'user_expressions': {}, 'allow_stdin': False,
                       'stop_on_error': True},
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
            out.append('ERR ' + str(msg['content'].get('ename')) + ':' +
                       str(msg['content'].get('evalue')))
        if msg.get('channel') == 'shell' and msg.get('msg_type') == 'execute_reply':
            break
    ws.close()
    try:
        requests.delete(BASE + '/api/kernels/' + kid, params={'token': TOKEN}, timeout=15)
    except Exception:
        pass
    return ''.join(out)


with open(r'[LOCAL-WORKSPACE]\teleagent\.temp\scnet_v8\evo_daemon.py', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')
api_put('/root/private_data/evo_daemon.py', b64)
print('uploaded')

print(run_code(
    "import subprocess\n"
    "print(subprocess.run(['bash','-lc',"
    "'pkill -f evo_daemon.py; sleep 3; "
    "cd /root/private_data && nohup python3 evo_daemon.py "
    "> evo_daemon_wrap.log 2>&1 & sleep 2; "
    "ps aux | grep evo_daemon | grep -v grep'],"
    "capture_output=True, text=True).stdout)", timeout=120))
