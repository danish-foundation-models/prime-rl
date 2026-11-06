"""Delete every sandbox with labels.run == argv[1]. Refuses while any of the user's jobs is running."""
import collections, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from ucloud_sandboxes_sdk.client import SandboxClient
label = sys.argv[1]
running = subprocess.run(["squeue", "--me", "-h", "-t", "RUNNING", "-o", "%i %j"], capture_output=True, text=True).stdout.strip()
assert not running, f"jobs running, not deleting: {running}"
c = SandboxClient.from_env()
ids = [s["id"] for s in c.list_sandboxes() if (s.get("labels") or {}).get("run") == label]
print("deleting", len(ids), "with run =", label)
def d(i):
    try:
        c.delete_sandbox(i); return "ok"
    except Exception as e:
        return type(e).__name__ + ": " + str(e)[:80]
with ThreadPoolExecutor(32) as ex:
    res = collections.Counter(ex.map(d, ids))
print(res.most_common(5))
print("total sandboxes left:", len(c.list_sandboxes()))
