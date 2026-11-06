"""Cancel the job right after its last checkpoint that fits before the wall-time limit.

Checkpoints are written every INTERVAL steps and nothing saves one at wall-time, so steps after the
last complete checkpoint are lost. After each checkpoint completes (trainer .metadata and orchestrator
progress.pt both written), estimate the time for the next INTERVAL steps plus a save; if that would not
finish before the job's end time, scancel the job.

Usage: stop_after_last_ckpt.py JOBID RUN_DIR ATTEMPT_LOG_DIR
"""

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

INTERVAL = 5
MARGIN_S = 10 * 60
SAFETY = 1.15

job, run_dir, log_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
trainer_log = log_dir / "trainer" / "node_0.log"
ansi = re.compile(r"\x1b\[[0-9;]*m")
step_re = re.compile(r"^(\d\d:\d\d:\d\d) +SUCCESS Step (\d+) \| +((?:\d+h )?(?:\d+m )?(?:\d+s)?) \|")
save_re = re.compile(r"^(\d\d:\d\d:\d\d) +INFO Saving checkpoint at step (\d+)")


def log(msg):
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def seconds(text):
    return sum(int(n) * {"h": 3600, "m": 60, "s": 1}[u] for n, u in re.findall(r"(\d+)([hms])", text))


def job_end():
    out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%e"], capture_output=True, text=True).stdout.strip()
    return datetime.fromisoformat(out) if out else None


def trainer_history():
    durations, saves = {}, {}
    for line in trainer_log.read_text(errors="replace").splitlines():
        line = ansi.sub("", line)
        if m := step_re.match(line):
            durations[int(m.group(2))] = seconds(m.group(3))
        elif m := save_re.match(line):
            saves[int(m.group(2))] = m.group(1)
    return durations, saves


def complete(step):
    path = run_dir / "checkpoints" / f"step_{step}"
    return (path / "trainer" / ".metadata").is_file() and (path / "orchestrator" / "progress.pt").is_file()


end = job_end()
log(f"watching job {job}, ends {end}")
handled = set()
while (end := job_end()) is not None:
    durations, saves = trainer_history()
    for step in sorted(s for s in saves if s % INTERVAL == 0 and s not in handled and complete(s)):
        handled.add(step)
        meta = run_dir / "checkpoints" / f"step_{step}" / "trainer" / ".metadata"
        started = datetime.combine(datetime.now().date(), datetime.strptime(saves[step], "%H:%M:%S").time())
        save_s = max(60.0, meta.stat().st_mtime - started.timestamp()) if started.timestamp() <= meta.stat().st_mtime else 300.0
        recent = [durations[s] for s in sorted(durations) if s <= step][-2 * INTERVAL:]
        per_step = sum(recent) / len(recent)
        need = SAFETY * INTERVAL * per_step + save_s + MARGIN_S
        left = (end - datetime.now()).total_seconds()
        log(f"checkpoint {step} complete (save {save_s:.0f}s); next {INTERVAL} steps + save need ~{need/60:.0f} min, "
            f"{left/60:.0f} min left (mean step {per_step/60:.1f} min over {len(recent)})")
        if need > left:
            log(f"step {step + INTERVAL} would not be saved before {end:%H:%M}; cancelling {job} after checkpoint {step}")
            subprocess.run(["scancel", job], check=True)
            sys.exit(0)
    time.sleep(30)
log(f"job {job} left the queue")
