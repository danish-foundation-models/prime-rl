# Loaded automatically by every Python process that has infer-opt on PYTHONPATH (API server,
# engine cores and workers inherit it). Does nothing unless INFER_OPT names patches.
import os

if os.environ.get("INFER_OPT"):
    try:
        import infer_opt

        infer_opt.apply()
    except Exception as e:  # never take the server down from here
        import sys

        print(f"[infer_opt] failed to apply patches: {e!r}", file=sys.stderr)
