"""Read-only: relay stats and rollout registrations grouped by label/age. Prints no credentials."""
import collections, json
from ucloud_sandboxes_sdk.relay import RelayWorkerClient as RelayClient
c = RelayClient.from_env()
s = c.stats()
print("stats:", json.dumps(s, default=str)[:1500])
r = c.list_rollouts()
print("registered rollouts:", len(r))
if r:
    print("keys:", sorted(r[0].keys()))
    cnt = collections.Counter(json.dumps({k: v for k, v in x.items() if k in ("labels", "run", "state", "phase", "status")}, sort_keys=True, default=str)[:150] for x in r)
    for k, v in cnt.most_common(10): print(v, k)
    ts = sorted(str(x.get("created_at") or x.get("registered_at") or "") for x in r); print("oldest", ts[0], "newest", ts[-1])
