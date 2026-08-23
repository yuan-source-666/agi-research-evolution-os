import json, glob
d = []
for f in sorted(glob.glob('/root/private_data/v8/scaled/shard_*.json')):
    x = json.load(open(f))
    if isinstance(x, dict):
        x = x.get('results') or x.get('groups') or []
    d += x
print('###JSON###' + json.dumps({'groups': d}, ensure_ascii=True))
