"""Exercise managed agent parking, relay wake and process state; delete only the probe."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from ucloud_sandboxes_sdk import AsyncRelayWorkerClient
from verifiers_ucloud.runtime import UCloudRuntime, UCloudRuntimeConfig


async def main(args):
    root = Path("/scratch/project_465002751/rasmus")
    for file in [root / "post/.env", root / "prime-rl/.env"]:
        load_dotenv(file, override=True)
    name = "super-parking-probe-" + uuid.uuid4().hex[:12]
    events = []

    def record(event, **values):
        events.append({"event": event, "time": time.time(), **values})
        args.output.write_text(json.dumps(events, indent=2))
        print(json.dumps(events[-1]), flush=True)

    runtime = UCloudRuntime(
        UCloudRuntimeConfig(
            image=args.image,
            workdir="/tmp",
            parkable=True,
            cpu=1,
            memory=2,
            disk=5,
            allow=[],
            guest_relay_url="http://gateway-live-ucloud-20260824a:8092",
            labels={"run": name, "purpose": "parking-validation"},
            ttl_seconds=1800,
        ),
        name=name,
    )
    worker = agent = None
    try:
        await runtime.start()
        record(
            "created",
            id=runtime.info.id,
            generation=runtime.info.sandbox_generation,
            parkable=runtime.info.sandbox_handle.record["spec"]["parkable"],
            managed_process=runtime.info.sandbox_handle.record["spec"]["managed_process"],
        )
        async with AsyncRelayWorkerClient.from_env(timeout_seconds=30, forward_timeout_seconds=600) as relay:
            async with relay.rollout_session(name, sandbox=runtime.info.sandbox_handle) as tunnel:
                public = urlsplit(tunnel.base_url)
                guest = urlsplit("http://gateway-live-ucloud-20260824a:8092")
                endpoint = urlunsplit((guest.scheme, guest.netloc, public.path, public.query, public.fragment)).rstrip(
                    "/"
                )
                requests = asyncio.Queue()

                async def handler(request):
                    release = asyncio.Event()
                    await requests.put((request, release))
                    await release.wait()
                    return {"ok": True}

                worker = asyncio.create_task(tunnel.run(handler=handler, poll_timeout_seconds=2, lease_seconds=300))
                program = """import os,json,urllib.request
from pathlib import Path
memory=os.urandom(32).hex()
Path('/tmp/park-sentinel').write_text(memory)
for i in range(2):
 body=json.dumps({'model':'parking-probe','messages':[{'role':'user','content':'test'}],'stream':False}).encode()
 req=urllib.request.Request(os.environ['ENDPOINT']+'/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=600) as response:
  assert json.load(response)['ok']
 assert Path('/tmp/park-sentinel').read_text()==memory
 print('resumed-cycle-'+str(i),flush=True)
"""
                agent = asyncio.create_task(
                    runtime.run_program(["python3", "-u", "-c", program], {"ENDPOINT": endpoint})
                )
                for cycle in range(2):
                    request, release = await asyncio.wait_for(requests.get(), 180)
                    assert request.sandbox_id == runtime.info.id
                    record(
                        "relay_request", cycle=cycle, sandbox_id=request.sandbox_id, reattachable=request.reattachable
                    )
                    deadline = time.monotonic() + 150
                    previous = None
                    while time.monotonic() < deadline:
                        row = await runtime._client_or_raise().get_sandbox(runtime.info.id)
                        state = row.get("state", row.get("status"))
                        if state != previous:
                            record("sandbox_state", cycle=cycle, state=state)
                            previous = state
                        if state == "parked":
                            break
                        if agent.done():
                            result = await agent
                            raise RuntimeError(
                                f"Agent exited before parking: {result.exit_code}: {result.stderr[-500:]}"
                            )
                        await asyncio.sleep(2)
                    else:
                        raise TimeoutError("Sandbox did not park during model wait")
                    release.set()
                    record("response_released", cycle=cycle)
                result = await asyncio.wait_for(agent, 180)
                assert result.exit_code == 0, result.stderr[-1000:]
                assert result.stdout.count("resumed-cycle-") == 2, result.stdout
                record("passed", stdout=result.stdout, cycles=2)
    finally:
        for task in (agent, worker):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (agent, worker) if t is not None), return_exceptions=True)
        # The runtime owns exactly this randomly named probe; never enumerate/delete others.
        await runtime.stop()
        record("cleanup_complete", id=name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default="python:3.11-slim")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main(args))
