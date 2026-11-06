"""Unregister relay rollouts registered before the live job started (argv[1] = epoch cutoff), oldest
first, at most argv[2] of them. Prints counts only, never tokens."""
import collections
import sys
from concurrent.futures import ThreadPoolExecutor

from ucloud_sandboxes_sdk.relay import RelayWorkerClient

cutoff, limit = float(sys.argv[1]), int(sys.argv[2])
c = RelayWorkerClient.from_env()
before = c.stats()
stale = sorted((x for x in c.list_rollouts() if x["registered_at"] < cutoff), key=lambda x: x["registered_at"])[:limit]


def unregister(x):
    try:
        c.unregister_rollout(x["rollout_id"], registration_token=x["registration_token"])
        return "ok"
    except Exception as e:
        return type(e).__name__ + ": " + str(e)[:100]


with ThreadPoolExecutor(16) as ex:
    print(collections.Counter(ex.map(unregister, stale)).most_common(4))
after = c.stats()
print("rollouts", before["rollouts"], "->", after["rollouts"],
      "| completed_bytes", before["completed_bytes"], "->", after["completed_bytes"])
