"""Check required provider settings by presence before submitting GPU work."""
import json
import os
import sys
from pathlib import Path

def uses_ucloud(value):
    if isinstance(value, dict):
        return value.get("type") == "ucloud" or any(uses_ucloud(v) for v in value.values())
    return isinstance(value, list) and any(uses_ucloud(v) for v in value)

root = Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit("Missing resolved config directory")
configs = list(root.rglob("*.json"))
if not configs:
    raise SystemExit("No resolved JSON configs found")
if any(uses_ucloud(json.loads(path.read_text())) for path in configs):
    names = ("UCLOUD_SANDBOX_URL", "UCLOUD_SANDBOX_API_TOKEN", "UCLOUD_RELAY_URL", "UCLOUD_RELAY_WORKER_TOKEN")
    missing = [name for name in names if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit("Missing UCloud settings: " + ", ".join(missing))
print("Provider credential presence check passed")
