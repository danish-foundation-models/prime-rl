"""Live UCloud gateway probes with r21's runtime settings; deletes only its own sandboxes.

group   identical managed sandboxes started together, group create on and off
agent   a managed primary program parks during relay model waits and resumes
acp     an interactive (open_process) program parks the same way, fed through stdin
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from ucloud_sandboxes_sdk import AsyncRelayWorkerClient
from verifiers_ucloud.runtime import UCloudRuntime, UCloudRuntimeConfig

ROOT = Path("/scratch/project_465002751/rasmus")
BUNDLE = ROOT / "artifacts/training/nemotron-super-r5-rl-scale-20260917/offline-compaction-20260922/python-0.tar.gz"
GUEST_RELAY = "http://10.36.101.16:8092"
IMAGE = "aweaiteam/scaleswe:beetbox_beets_pr3661"

MODEL_CALLS = """
import json, os, urllib.request
from pathlib import Path
def call():
    body = json.dumps({"model": "probe", "messages": [{"role": "user", "content": "test"}], "stream": False}).encode()
    request = urllib.request.Request(os.environ["ENDPOINT"] + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        assert json.load(response)["ok"]
"""


class Log:
    def __init__(self, path: Path):
        self.path, self.events = path, []

    def __call__(self, event, **values):
        self.events.append({"event": event, "time": round(time.time(), 1), **values})
        self.path.write_text(json.dumps(self.events, indent=2))
        print(json.dumps(self.events[-1]), flush=True)


def runtime_config(label: str, **overrides) -> UCloudRuntimeConfig:
    settings = dict(
        image=IMAGE, image_reference_type="name", workdir="/tmp", cpu=1, memory=2, disk=5,
        allow=[], guest_relay_url=GUEST_RELAY, managed_agent=True, park_interactive=True,
        repair_loopback_hosts=True, offline_python_bundle=BUNDLE,
        labels={"run": label, "purpose": "r21-runtime-probe"}, ttl_seconds=1800,
    )
    settings.update(overrides)
    return UCloudRuntimeConfig(**settings)


def managed_record(runtime) -> dict:
    record = runtime.sandbox.record
    spec = record.get("spec", {})
    return {"id": runtime.info.id, "generation": record.get("generation"),
            "parkable": spec.get("parkable"), "managed_process": spec.get("managed_process")}


async def probe_group(log: Log, count: int) -> bool:
    passed = True
    for group_create in (True, False):
        label = f"r21-probe-group-{'on' if group_create else 'off'}-{uuid.uuid4().hex[:8]}"
        config = runtime_config(label, group_create=group_create)
        runtimes = [UCloudRuntime(config, name=f"{label}-{i}") for i in range(count)]
        started = time.monotonic()
        try:
            outcomes = await asyncio.gather(*(r.start() for r in runtimes), return_exceptions=True)
            seconds = round(time.monotonic() - started, 1)
            members = []
            for runtime, outcome in zip(runtimes, outcomes):
                if isinstance(outcome, BaseException):
                    members.append({"error": f"{type(outcome).__name__}: {outcome}"[:400]})
                    continue
                result = await runtime.run(["sh", "-c", "echo ok; getent hosts localhost | head -1"], {})
                members.append({**managed_record(runtime), "exit": result.exit_code, "stdout": result.stdout.strip()})
            ok = all("error" not in m and m["parkable"] and m["managed_process"]
                     and isinstance(m["generation"], int) and m["generation"] >= 1
                     and m["exit"] == 0 and m["stdout"].startswith("ok") for m in members)
            passed &= ok
            log("group", group_create=group_create, start_seconds=seconds, passed=ok, members=members)
        finally:
            await asyncio.gather(*(r.stop() for r in runtimes), return_exceptions=True)
            log("group_cleanup", label=label)
    return passed


async def parking_cycles(runtime, tunnel, log: Log, program_task, cycles=2, hold_seconds=30) -> bool:
    """Hold each model call, recording the sandbox states seen while it waits.

    The gateway pauses a waiting managed sandbox in place and parks (hibernates)
    it only under node memory pressure, so a held call must simply be answered
    and the program must resume with its memory intact.
    """
    requests = asyncio.Queue()

    async def handler(request):
        release = asyncio.Event()
        await requests.put((request, release))
        await release.wait()
        return {"ok": True}

    worker = asyncio.create_task(tunnel.run(handler=handler, poll_timeout_seconds=2, lease_seconds=300))
    try:
        for cycle in range(cycles):
            request, release = await asyncio.wait_for(requests.get(), 300)
            log("relay_request", cycle=cycle, sandbox_id=request.sandbox_id, reattachable=request.reattachable)
            states, deadline = [], time.monotonic() + hold_seconds
            while time.monotonic() < deadline and not program_task.done():
                row = await runtime._client_or_raise().get_sandbox_status(runtime.info.id)
                state = (row or {}).get("state")
                if not states or states[-1] != state:
                    states.append(state)
                await asyncio.sleep(2)
            log("held", cycle=cycle, seconds=hold_seconds, states=states, program_alive=not program_task.done())
            release.set()
            if program_task.done():
                return False
        return True
    finally:
        await asyncio.sleep(1)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


async def probe_agent(log: Log) -> bool:
    name = f"r21-probe-agent-{uuid.uuid4().hex[:8]}"
    runtime = UCloudRuntime(runtime_config(name, group_create=False), name=name)
    program = MODEL_CALLS + """
memory = os.urandom(16).hex(); Path("/tmp/park-sentinel").write_text(memory)
for i in range(2):
    call(); assert Path("/tmp/park-sentinel").read_text() == memory
    print("resumed-cycle-" + str(i), flush=True)
"""
    agent = None
    try:
        await runtime.start()
        log("agent_created", **managed_record(runtime))
        async with AsyncRelayWorkerClient.from_env(timeout_seconds=30, forward_timeout_seconds=600) as relay, \
                relay.rollout_session(name, sandbox=runtime.sandbox) as tunnel:
            endpoint = runtime.host_url(tunnel.base_url).rstrip("/")
            log("agent_endpoint", guest=endpoint.split("/")[2])
            # The sandbox image has python3; the program needs only the standard library.
            agent = asyncio.create_task(runtime.run_program(["python3", "-u", "-c", program], {"ENDPOINT": endpoint}))
            held = await parking_cycles(runtime, tunnel, log, agent)
            result = await asyncio.wait_for(agent, 300)
        ok = held and result.exit_code == 0 and result.stdout.count("resumed-cycle-") == 2
        log("agent_result", passed=ok, exit=result.exit_code, stdout=result.stdout[-300:], stderr=result.stderr[-500:])
        return ok
    finally:
        if agent is not None and not agent.done():
            agent.cancel()
            await asyncio.gather(agent, return_exceptions=True)
        await runtime.stop()
        log("agent_cleanup", label=name)


async def probe_acp(log: Log) -> bool:
    from verifiers.v1.acp import ACP_SOURCE

    name = f"r21-probe-acp-{uuid.uuid4().hex[:8]}"
    runtime = UCloudRuntime(runtime_config(name, group_create=False), name=name)
    program = MODEL_CALLS + """
import sys
memory = os.urandom(16).hex(); Path("/tmp/park-sentinel").write_text(memory)
for i in range(2):
    line = sys.stdin.readline().strip()
    call(); assert Path("/tmp/park-sentinel").read_text() == memory
    print("answered-" + line, flush=True)
"""
    process = None
    try:
        await runtime.start()
        log("acp_created", **managed_record(runtime))
        # Stage the pinned offline Python through ACP's own allowlisted script.
        interpreter = (await runtime.prepare_uv_script(ACP_SOURCE, {}, activate=False))[0]
        async with AsyncRelayWorkerClient.from_env(timeout_seconds=30, forward_timeout_seconds=600) as relay, \
                relay.rollout_session(name, sandbox=runtime.sandbox) as tunnel:
            endpoint = runtime.host_url(tunnel.base_url).rstrip("/")
            process = await runtime.open_process([interpreter, "-I", "-c", program], {"ENDPOINT": endpoint})
            log("acp_process", kind=type(process).__name__)
            output = bytearray()

            async def drain():
                async for chunk in process.stdout:
                    output.extend(chunk)

            reader = asyncio.create_task(drain())
            waiter = asyncio.create_task(process.wait())
            for line in (b"first\n", b"second\n"):
                await process.write(line)
            held = await parking_cycles(runtime, tunnel, log, waiter)
            exit_code = await asyncio.wait_for(waiter, 300)
            await asyncio.wait_for(reader, 60)
        text = output.decode(errors="replace")
        ok = (held and exit_code == 0 and type(process).__name__ == "ManagedProcess"
              and "answered-first" in text and "answered-second" in text)
        log("acp_result", passed=ok, exit=exit_code, stdout=text[-300:])
        return ok
    finally:
        if process is not None:
            try:
                await process.kill()
            except Exception:
                pass
        await runtime.stop()
        log("acp_cleanup", label=name)


async def main(args):
    for env_file in (ROOT / "post/.env", ROOT / "prime-rl/.env"):
        load_dotenv(env_file, override=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.probes:
        log = Log(args.output / f"{name}.json")
        try:
            results[name] = await {"group": lambda: probe_group(log, args.count),
                                   "agent": lambda: probe_agent(log),
                                   "acp": lambda: probe_acp(log)}[name]()
        except Exception as exc:
            log("error", error=f"{type(exc).__name__}: {exc}"[:1500])
            results[name] = False
    (args.output / "summary.json").write_text(json.dumps(results, indent=2))
    print("SUMMARY", json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", nargs="+", choices=("group", "agent", "acp"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
