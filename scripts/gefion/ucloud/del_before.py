"""Delete every sandbox created before argv[1] (ISO 8601 UTC), whatever its label: the leftovers of
a cancelled job, while a newer job with the same label is already creating its own."""
import collections, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from ucloud_sandboxes_sdk.client import SandboxClient
cutoff = datetime.fromisoformat(sys.argv[1])
c = SandboxClient.from_env()
sb = c.list_sandboxes()
old = [s["id"] for s in sb if datetime.fromisoformat(s["created_at"]) < cutoff]
print(f"{len(sb)} sandboxes, deleting {len(old)} created before {cutoff.isoformat()}")
def d(i):
    try:
        c.delete_sandbox(i); return "ok"
    except Exception as e:
        return type(e).__name__ + ": " + str(e)[:80]
with ThreadPoolExecutor(32) as ex:
    print(collections.Counter(ex.map(d, old)).most_common(5))
left = c.list_sandboxes()
print("left:", len(left), "of which older than cutoff:", sum(datetime.fromisoformat(s["created_at"]) < cutoff for s in left))
