"""Check managed bash relay completion and state, optionally requiring cold parking."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from aiohttp import web
from dotenv import load_dotenv
from verifiers.v1.agent import Agent
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.task import Task, TaskData
from verifiers_ucloud import UCloudInterception, UCloudInterceptionConfig, UCloudRuntime, UCloudRuntimeConfig


async def main(args):
    root = Path("/scratch/project_465002751/rasmus")
    for file in [root / "post/.env", root / "prime-rl/.env"]:
        load_dotenv(file, override=True)
    name = "super-park-harness-" + uuid.uuid4().hex[:12]
    events = []

    def record(event, **data):
        events.append({"event": event, "time": time.time(), **data})
        args.output.write_text(json.dumps(events, indent=2))
        print(json.dumps(events[-1]), flush=True)

    runtime = UCloudRuntime(
        UCloudRuntimeConfig(
            image=args.image,
            image_reference_type=args.image_type,
            parkable=True,
            workdir="/tmp",
            cpu=2,
            memory=4,
            disk=20,
            allow=[],
            guest_relay_url=args.guest_relay_url,
            ttl_seconds=1800,
            offline_python_bundle=root
            / "artifacts/training/nemotron-super-r5-rl-scale-20260917/offline-compaction-20260922/python-0.tar.gz",
            labels={"run": name, "purpose": "parking-harness-validation"},
        ),
        name=name,
    )
    interception = UCloudInterception(
        UCloudInterceptionConfig(guest_relay_url=args.guest_relay_url)
    )
    calls = 0

    async def completion(request):
        nonlocal calls
        body = await request.json()
        calls += 1
        assert calls <= 2, "Unexpected retry/duplicate model request"
        deadline = time.monotonic() + (120 if args.require_park else 5)
        while time.monotonic() < deadline:
            state = await runtime._client_or_raise().get_sandbox(runtime.info.id)
            if state.get("state") == "parked":
                break
            await asyncio.sleep(1)
        else:
            if args.require_park or state.get("state") != "running":
                raise RuntimeError(f"Unexpected model-wait state: {state.get('state')}")
        record("model_wait", call=calls, state=state.get("state"),
               warm_retained=state.get("state") == "running")
        if calls == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "probe-tool",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {
                                    "command": "printf parking-tool-ok > /tmp/harness-parking-sentinel; cat /tmp/harness-parking-sentinel"
                                }
                            ),
                        },
                    }
                ],
            }
        else:
            assert any(m.get("role") == "tool" and "parking-tool-ok" in str(m.get("content")) for m in body["messages"])
            message = {"role": "assistant", "content": "PARKING_HARNESS_OK"}
        return web.json_response(
            {
                "id": "probe-" + str(calls),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "parking-probe",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls == 1 else "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    runner = web.AppRunner(app)
    try:
        await runtime.start()
        record("created", id=runtime.info.id, parkable=runtime.info.parkable)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        await interception.start()
        config = AgentConfig.model_validate(
            {
                "harness": {"id": "bash"},
                "runtime": runtime.config,
                "model": "parking-probe",
                "client": {
                    "type": "eval",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key_var": "PARKING_PROBE_UNUSED_KEY",
                },
                "max_turns": 3,
                "timeout": {"setup": 300, "rollout": 400},
            }
        )
        task = Task(
            TaskData(prompt="Run the requested shell command and report completion.", workdir="/tmp", network_allow=[])
        )
        async with Agent(config, interception=interception) as agent:
            trace = await agent.run(task, runtime=runtime)
        args.output.with_suffix(".trace.json").write_text(trace.model_dump_json(indent=2))
        assert trace.ok, str(trace.errors)
        assert calls == 2, calls
        assert (await runtime.read("/tmp/harness-parking-sentinel")) == b"parking-tool-ok"
        assert "sandbox_handle" not in trace.model_dump_json()
        record("passed", calls=calls, stop_condition=trace.stop_condition, tool_state_preserved=True)
    finally:
        await interception.stop()
        await runtime.stop()
        await runner.cleanup()
        record("cleanup_complete", id=name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default="python:3.11-slim")
    p.add_argument("--image-type", choices=["name", "registry"], default="registry")
    p.add_argument("--guest-relay-url", default="http://gateway-live-ucloud-20260824a:8092")
    p.add_argument("--require-park", action="store_true", help="Require cold parking; omit for warm-retention deployments")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main(args))
