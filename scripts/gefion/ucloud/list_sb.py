import collections, json, sys
from ucloud_sandboxes_sdk.client import SandboxClient
c = SandboxClient.from_env()
sb = c.list_sandboxes()
print("total", len(sb))
print("keys", sorted(sb[0].keys()) if sb else None)
cnt = collections.Counter()
for s in sb:
    labels = s.get("labels") or (s.get("spec") or {}).get("labels") or {}
    cnt[(json.dumps(labels, sort_keys=True)[:200], s.get("state") or s.get("status"))] += 1
for k, v in cnt.most_common(30): print(v, k)
