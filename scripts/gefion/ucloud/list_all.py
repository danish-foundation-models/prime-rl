import collections, json
from ucloud_sandboxes_sdk.client import SandboxClient
c = SandboxClient.from_env()
sb = c.list_sandboxes()
print("total", len(sb))
cnt = collections.Counter((json.dumps(s.get("labels") or {}, sort_keys=True), s.get("state")) for s in sb)
for k, v in cnt.most_common(20): print(v, k)
ts = sorted(s.get("created_at") or "" for s in sb)
if ts: print("created from", ts[0], "to", ts[-1])
img = collections.Counter((s.get("image") or "")[:60] for s in sb)
print(img.most_common(5))
