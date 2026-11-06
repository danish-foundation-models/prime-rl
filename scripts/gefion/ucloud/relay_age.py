import collections, json, datetime as dt
from ucloud_sandboxes_sdk.relay import RelayWorkerClient
RS = 1793453388  # 534519 start, 14:29:48 CET
r = RelayWorkerClient.from_env().list_rollouts()
b = collections.Counter()
for x in r:
    t = x["registered_at"]; d = dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=1)))
    b[("live" if t >= RS else "stale", d.strftime("%m-%d %H:00"))] += 1
for k in sorted(b): print(k, b[k])
print("metadata examples:", [json.dumps(x["metadata"], default=str)[:200] for x in r[:3]])
print("stale:", sum(x["registered_at"] < RS for x in r), "live:", sum(x["registered_at"] >= RS for x in r))
