"""Watch an already-submitted pilot and reserve CPU/build capacity when it starts.

This observer never resubmits jobs, changes task pools, or deletes sandboxes.
Opt-in --stop-on-infra-failure cancels only its specified job on fatal component
errors, sustained infrastructure failures, or no useful progress. Confirmed
image-build failures are task-local exclusions, not global stop signals.
It records the first completed optimizer step and keeps watching until Slurm ends.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from collections import Counter, deque
from pathlib import Path
import re
import subprocess
import time


def tail(path, limit=1024*1024):
    if not path.exists():
        return ''
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size-limit))
        return handle.read().decode(errors='replace')


def fatal_component_errors(text):
    """Recognize explicit component-fatal logger records, not generated trace text."""
    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return [line for line in plain.splitlines() if re.match(
        r"^\d{2}:\d{2}:\d{2}\s+ERROR\s+Fatal error in run_(?:orchestrator|trainer)\b", line
    )]


class ProgressWatch:
    """Detect absence of useful work without treating long active tasks as stuck."""

    def __init__(self, now, *, min_generation_tps=0, throughput_window=600, throughput_warmup=1200):
        self.started_at = now
        self.last_progress_at = now
        self.active = False
        self.counters = {}
        self.min_generation_tps = min_generation_tps
        self.throughput_window = throughput_window
        self.throughput_warmup = throughput_warmup
        self.generation_samples = deque()
        self.first_generation_at = None
        self.generation_tps = None
        self.policy_update_pending = False
        self.generation_keys = set()

    def observe(self, now, *, successful_episodes, trainer_step, metrics=()):
        values = {"successful_episodes": successful_episodes, "trainer_step": trainer_step}
        for row in metrics:
            if "dispatcher/policy_update_pending" in row:
                self.policy_update_pending = bool(row["dispatcher/policy_update_pending"])
                if self.policy_update_pending:
                    self.generation_samples.clear()
            for name, value in row.items():
                if re.fullmatch(r"inference/server[^/]+/(?:generation|prompt)_tokens_total", name):
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        values[name] = value
        for name, value in values.items():
            if name.endswith("/generation_tokens_total") and value < self.counters.get(name, 0):
                self.generation_samples.clear()
            if value > self.counters.get(name, 0):
                self.active = True
                self.last_progress_at = now
            self.counters[name] = value
        keys = {key for key in self.counters if key.endswith("/generation_tokens_total")}
        total = sum(self.counters[key] for key in keys)
        if keys != self.generation_keys:
            self.generation_samples.clear()
            self.generation_keys = keys
        if total > 0 and self.first_generation_at is None:
            self.first_generation_at = now
        self.generation_tps = None
        if keys and not self.policy_update_pending:
            self.generation_samples.append((now, total))
            while len(self.generation_samples) > 2 and self.generation_samples[1][0] <= now - self.throughput_window:
                self.generation_samples.popleft()
            first_at, first_total = self.generation_samples[0]
            if now - first_at >= self.throughput_window:
                self.generation_tps = max(0, total - first_total) / (now - first_at)

    def failure(self, now, *, idle_seconds, startup_seconds, training_complete=False):
        if self.active and now - self.last_progress_at >= idle_seconds:
            return "no token, successful rollout, or optimizer progress"
        if not self.active and now - self.started_at >= startup_seconds:
            return "startup produced no token, successful rollout, or optimizer progress"
        if (not training_complete and self.min_generation_tps > 0 and self.first_generation_at is not None
            and now - self.first_generation_at >= self.throughput_warmup
            and self.generation_tps is not None and self.generation_tps < self.min_generation_tps):
            return f"sustained low generation throughput: {self.generation_tps:.1f} < {self.min_generation_tps:.1f} tokens/s"
        return None


def read_metric_rows(path, offset, *, after_time=0):
    rows = []
    if path.exists():
        with path.open("rb") as handle:
            handle.seek(offset)
            while True:
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                if row.get("time", 0) >= after_time:
                    rows.append(row)
                offset = handle.tell()
    return rows, offset


def trace_errors(run_dir, row):
    """Read only error records; generated/user text is not an error signal."""
    chunk = row.get("chunk")
    offset = row.get("offset")
    if not isinstance(chunk, int) or chunk < 0 or not isinstance(offset, int) or offset < 0:
        return []
    path = run_dir / "monitors/file/traces/stream" / f"{chunk:05d}.jsonl"
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            episode = json.loads(handle.readline())
    except (OSError, ValueError):
        return []
    errors = [error for trace in episode.get("traces") or [] if not trace.get("ok") for error in trace.get("errors") or []]
    if not errors:
        errors = list(episode.get("errors") or [])
    return errors


def trace_failure_kind(run_dir, row):
    return classify_trace_errors(trace_errors(run_dir, row))


def classify_trace_errors(errors):
    if not errors:
        return "unclassified_errors"
    if errors and all(
        error.get("type") == "ImageBuildFailure" or (
            error.get("type") == "SandboxError" and str(error.get("message", "")).startswith((
                "ucloud sandbox provisioning failed: Cached image build failure for ",
                "ucloud sandbox provisioning failed: Image preparation failed for ",
            ))
        ) for error in errors
    ):
        return "unavailable_task"
    if errors and all(re.search(r"\bnode_lost\b", str(error.get("message", ""))) or error.get("type") == "SandboxNodeLost" for error in errors):
        return "node_lost"
    return "other_errors"


def failure_incident(errors, row):
    """Deduplicate only explicitly typed, shared build-status failures."""
    ids = set()
    for error in errors:
        match = re.match(r"Image build status unavailable \[incident=([0-9a-f]{32})\]", str(error.get("message", "")))
        if error.get("type") != "ImageBuildPollingError" or match is None:
            return ("episode", row["id"])
        ids.add(match[1])
    return ("image_build", *sorted(ids)) if ids else ("episode", row["id"])


def failure_window(recent):
    affected = sum(not ok for ok, _ in recent)
    incidents = len({incident for ok, incident in recent if not ok})
    return {"eligible_episodes": len(recent), "failed_episodes": affected, "failure_incidents": incidents}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',type=int,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--logs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--builder-prepare-id',required=True)
    parser.add_argument('--capacity-prepare-id',required=True)
    parser.add_argument('--root',type=Path,default=Path('/scratch/project_465002751/rasmus'))
    parser.add_argument('--poll-seconds',type=float,default=45)
    parser.add_argument('--stop-on-infra-failure',action='store_true',help='Cancel on sustained infrastructure failures or no useful progress; image-build exclusions and rewards do not count')
    parser.add_argument("--tolerate-node-lost", action="store_true", help="Exclude terminal node-loss episodes from the stop window; retain counts and errors")
    parser.add_argument("--max-failed-episodes", type=int, default=15, help="Independent-failure threshold in the last 50 eligible episodes; shared build incidents count once")
    parser.add_argument("--max-no-progress-seconds", type=float, default=600,
                        help="After useful work starts, stop after this many seconds without token/rollout/optimizer progress")
    parser.add_argument("--max-startup-seconds", type=float, default=3600,
                        help="Initial startup budget before any useful progress")
    parser.add_argument("--min-generation-tps", type=float, default=500,
                        help="Aggregate generation-rate floor for this eight-replica run; 0 disables this guard")
    parser.add_argument("--throughput-window-seconds", type=float, default=600)
    parser.add_argument("--throughput-warmup-seconds", type=float, default=1200)
    args=parser.parse_args()
    if args.min_generation_tps < 0 or min(args.throughput_window_seconds, args.throughput_warmup_seconds) <= 0:
        parser.error("Throughput floor must be nonnegative and window/warmup positive")
    if min(args.max_no_progress_seconds, args.max_startup_seconds) <= 0:
        parser.error("Progress timeout budgets must be positive")
    if not 1 <= args.max_failed_episodes <= 50:parser.error("max-failed-episodes must be between 1 and 50")
    if args.poll_seconds < 10:parser.error('poll-seconds must be at least10')
    if any(f'-{args.job_id}-' not in value for value in [args.builder_prepare_id,args.capacity_prepare_id]):
        parser.error('Reservation IDs must be bound to this Slurm job ID')
    args.output.mkdir(parents=True,exist_ok=True)
    first_step=None;last_capacity=0;started=False
    trace_offset=0;trace_counts=Counter();source_counts=Counter();recent=deque(maxlen=50);stop_requested=False
    fatal_errors = []
    progress_watch = None
    run_start_epoch = None
    metric_offset = 0
    while True:
        now=datetime.now(timezone.utc).isoformat()
        try:
            result=subprocess.run(['squeue','-h','-j',str(args.job_id),'-o','%T|%M|%S|%R'],capture_output=True,text=True,timeout=30,check=True,env={**os.environ, 'TZ': 'UTC'})
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            print(json.dumps({'time':now,'poll_error':str(error)}),flush=True)
            time.sleep(args.poll_seconds)
            continue
        queue=result.stdout.strip();state=queue.split('|')[0] if queue else 'ENDED'
        snapshot={'time':now,'job_id':args.job_id,'slurm':queue,'state':state,'run_dir':str(args.run_dir)}
        if state in ['RUNNING','CONFIGURING']:
            started=True
            if progress_watch is None:
                progress_watch = ProgressWatch(time.monotonic(), min_generation_tps=args.min_generation_tps,
                    throughput_window=args.throughput_window_seconds, throughput_warmup=args.throughput_warmup_seconds)
                start_text = queue.split('|')[2]
                run_start_epoch = (datetime.fromisoformat(start_text).replace(tzinfo=timezone.utc).timestamp()
                    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", start_text)
                    else datetime.now(timezone.utc).timestamp())
            if time.monotonic()-last_capacity>1800:
                try:
                    scripts=args.root / Path(__file__).resolve().parent.relative_to(args.root.resolve())
                    command=[str(scripts/'lumi_run_in_container.sh'),'python',str(scripts/'lumi_prepare_pilot_capacity.py'),'--root',str(args.root),'--job-id',str(args.job_id),'--builder-prepare-id',args.builder_prepare_id,'--capacity-prepare-id',args.capacity_prepare_id,'--output',str(args.output/'latest-reservations.json')]
                    with (args.output/'capacity.log').open('a') as log:
                        subprocess.run(command,stdout=log,stderr=log,timeout=300,check=True)
                    last_capacity=time.monotonic()
                except Exception as error:
                    # A transient provider error must not stop observation of the job.
                    snapshot['reservation_error']=f'{type(error).__name__}: {error}'
        trainer=tail(args.logs/'trainer/node_0.log')
        steps=re.findall(r'Step (\d+) \|[^\n]*?\| Loss [^\n]+',trainer)
        lines=[line for line in trainer.splitlines() if re.search(r'Step \d+ \|.*\| Loss ',line)]
        if steps and first_step is None:
            loss=float(re.search(r'\| Loss ([^ ]+)',lines[0]).group(1))
            first_step={'time':now,'step':int(steps[0]),'line':lines[0],'finite_loss':math.isfinite(loss)}
            (args.output/'first-optimizer-update.json').write_text(json.dumps(first_step,indent=2))
        snapshot['first_optimizer_update']=first_step
        snapshot['latest_step_line']=lines[-1] if lines else None
        orchestrator=tail(args.logs/'orchestrator.log')
        snapshot['orchestrator_tail']=orchestrator.splitlines()[-8:]
        fatal_errors = list(dict.fromkeys([*fatal_errors, *fatal_component_errors(orchestrator), *fatal_component_errors(trainer)]))
        snapshot['fatal_component_errors'] = fatal_errors
        snapshot['env_errors']={}
        for path in (args.logs/'envs/train').glob('*.log'):
            errors=[line for line in tail(path,256*1024).splitlines() if any(word in line for word in ['ERROR','RuntimeError','SandboxError','Traceback'])]
            if errors:snapshot['env_errors'][path.stem]=errors[-5:]
        index=args.run_dir/'monitors/file/traces/stream.index.jsonl'
        if index.exists():
            with index.open('rb') as handle:
                handle.seek(trace_offset)
                while True:
                    line=handle.readline()
                    if not line or not line.endswith(b'\n'):
                        break
                    row=json.loads(line);trace_offset=handle.tell()
                    if run_start_epoch is None or row.get('arrival', 0) < run_start_epoch:
                        continue
                    errors = [] if row.get('ok') else trace_errors(args.run_dir, row)
                    kind = 'ok' if row.get('ok') else classify_trace_errors(errors)
                    if kind != 'unavailable_task' and not (args.tolerate_node_lost and kind == 'node_lost'):
                        recent.append((bool(row.get('ok')), failure_incident(errors, row)))
                    trace_counts['completed']+=1
                    trace_counts['ok' if row.get('ok') else 'errors']+=1
                    if not row.get('ok'):
                        trace_counts[kind]+=1
                    source_counts[str(row.get('env'))]+=1
        stalled = None
        if progress_watch is not None:
            metric_rows, metric_offset = read_metric_rows(args.run_dir/'monitors/file/metrics.jsonl', metric_offset,
                                                          after_time=run_start_epoch)
            monotonic_now = time.monotonic()
            progress_watch.observe(monotonic_now, successful_episodes=trace_counts['ok'],
                                   trainer_step=int(steps[-1]) if steps else 0, metrics=metric_rows)
            trainer_done = args.run_dir/'launcher/.trainer.done'
            training_complete = (run_start_epoch is not None and trainer_done.exists()
                                 and trainer_done.stat().st_mtime >= run_start_epoch)
            stalled = progress_watch.failure(monotonic_now, idle_seconds=args.max_no_progress_seconds,
                                             startup_seconds=args.max_startup_seconds,
                                             training_complete=training_complete)
            snapshot['progress_health'] = {'active': progress_watch.active,
                'seconds_without_progress': monotonic_now - progress_watch.last_progress_at,
                'generation_tps': progress_watch.generation_tps,
                'min_generation_tps': 0 if training_complete else args.min_generation_tps,
                'training_complete': training_complete,
                'policy_update_pending': progress_watch.policy_update_pending,
                'failure': stalled}
        snapshot['failure_window'] = window = failure_window(recent)
        if args.stop_on_infra_failure and not stop_requested and state == 'RUNNING' and (fatal_errors or stalled or (len(recent) == 50 and window['failure_incidents'] >= args.max_failed_episodes)):
            reason={'time':now,'job_id':args.job_id,'failed':window['failed_episodes'],'failure_incidents':window['failure_incidents'],'window':50,'reason':'sustained infrastructure failures; image exclusions, tolerated node loss and rewards are not used' if args.tolerate_node_lost else 'sustained infrastructure failures; image exclusions and rewards are not used'}
            reason['excluded_image_failures'] = trace_counts['unavailable_task']
            if stalled:
                reason.update(reason=stalled, progress_health=snapshot['progress_health'])
            if fatal_errors:
                reason.update(reason='fatal training component error', errors=fatal_errors)
            (args.output/'automatic-stop.json').write_text(json.dumps(reason,indent=2))
            subprocess.run(['scancel',str(args.job_id)],check=True,timeout=30)
            stop_requested=True
            print(json.dumps({'automatic_stop':reason}),flush=True)
        snapshot['stop_requested']=stop_requested
        snapshot['episodes']=dict(trace_counts)
        snapshot['episodes_by_source']=dict(source_counts)
        snapshot['started']=started
        temp=args.output/'status.tmp';temp.write_text(json.dumps(snapshot,indent=2));temp.replace(args.output/'status.json')
        summary={'time':now,'state':state,'first_optimizer_update':first_step['step'] if first_step else None,'envs_with_errors':list(snapshot['env_errors']),'episodes':dict(trace_counts)}
        print(json.dumps(summary),flush=True)
        if state=='ENDED':
            accounting=subprocess.run(['sacct','-j',str(args.job_id),'--format=JobID,State,ExitCode,Elapsed','--noheader'],capture_output=True,text=True,timeout=30)
            (args.output/'slurm-final.txt').write_text(accounting.stdout)
            scripts=args.root / Path(__file__).resolve().parent.relative_to(args.root.resolve())
            release=[str(scripts/'lumi_run_in_container.sh'),'python',str(scripts/'lumi_prepare_pilot_capacity.py'),'--release','--root',str(args.root),'--job-id',str(args.job_id),'--builder-prepare-id',args.builder_prepare_id,'--capacity-prepare-id',args.capacity_prepare_id,'--output',str(args.output/'released-reservations.json')]
            with (args.output/'capacity.log').open('a') as log:
                result=subprocess.run(release,stdout=log,stderr=log,timeout=300)
            print(f'Slurm ended; reservation release exit={result.returncode}. Sandboxes were not deleted.',flush=True)
            break
        time.sleep(args.poll_seconds)

if __name__=='__main__':main()
