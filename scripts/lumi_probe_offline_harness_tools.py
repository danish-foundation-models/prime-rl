"""Validate offline coding tools through real parkable ACP harnesses."""

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
    name = "super-park-" + args.harness + "-" + uuid.uuid4().hex[:12]
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
            disk=5,
            allow=[],
            guest_relay_url="http://gateway-live-ucloud-20260824a:8092",
            ttl_seconds=1800,
            offline_python_bundle=root / "artifacts/training/nemotron-super-r5-rl-scale-20260917/harnesses-20260920/offline-python.tar.gz",
            offline_harness_bundle=(root / f"artifacts/training/nemotron-super-r5-rl-scale-20260917/harnesses-20260920/search-tools-v1/{args.harness}-assets.tar.gz") if args.harness != "bash" else None,
            labels={"run": name, "purpose": "parking-harness-validation"},
        ),
        name=name,
    )
    interception = UCloudInterception(
        UCloudInterceptionConfig(guest_relay_url="http://gateway-live-ucloud-20260824a:8092")
    )
    calls = 0
    filename = "/tmp/harness-tools-probe.txt"
    if args.harness == "opencode":
        plan = [
            ("write", {"filePath": filename, "content": "SEARCH_SENTINEL"}, None),
            ("glob", {"pattern": "harness-tools-probe.txt", "path": "/tmp"}, "harness-tools-probe.txt"),
            ("grep", {"pattern": "SEARCH_SENTINEL", "path": "/tmp", "include": "harness-tools-probe.txt"}, "SEARCH_SENTINEL"),
            ("read", {"filePath": filename}, "SEARCH_SENTINEL"),
            ("edit", {"filePath": filename, "oldString": "SEARCH_SENTINEL", "newString": "EDITED_SENTINEL"}, None),
            ("bash", {"command": f"rg --version && fd --version && cat {filename}"}, "EDITED_SENTINEL"),
        ]
    elif args.harness == "pi":
        plan = [
            ("write", {"path": filename, "content": "SEARCH_SENTINEL"}, None),
            ("bash", {"command": "fd --glob harness-tools-probe.txt /tmp"}, "harness-tools-probe.txt"),
            ("bash", {"command": f"rg SEARCH_SENTINEL {filename}"}, "SEARCH_SENTINEL"),
            ("read", {"path": filename}, "SEARCH_SENTINEL"),
            ("edit", {"path": filename, "edits": [{"oldText": "SEARCH_SENTINEL", "newText": "EDITED_SENTINEL"}]}, None),
            ("bash", {"command": f"rg --version && fd --version && cat {filename}"}, "EDITED_SENTINEL"),
        ]
    else:
        plan = [
            ("ipython", {"code": "from pathlib import Path; import subprocess; p=Path('/tmp/harness-tools-probe.txt'); p.write_text('SEARCH_SENTINEL'); print('WRITTEN')"}, "WRITTEN"),
            ("ipython", {"code": "print(subprocess.check_output(['fd','--glob','harness-tools-probe.txt','/tmp'],text=True))"}, "harness-tools-probe.txt"),
            ("ipython", {"code": "print(subprocess.check_output(['rg','SEARCH_SENTINEL',str(p)],text=True))"}, "SEARCH_SENTINEL"),
            ("ipython", {"code": "print(p.read_text()); p.write_text(p.read_text().replace('SEARCH_SENTINEL','EDITED_SENTINEL')); print(p.read_text())"}, "EDITED_SENTINEL"),
        ]

    async def completion(request):
        nonlocal calls
        body = await request.json()
        (args.output.with_suffix(f".request-{calls+1}.json")).write_text(json.dumps(body,indent=2))
        calls += 1
        assert calls <= len(plan) + 1, "Unexpected retry/duplicate request"
        if calls > 1:
            results = [m for m in body["messages"] if m.get("role") == "tool"]
            assert results, "Missing tool result"
            result = str(results[-1].get("content"))
            expected = plan[calls - 2][2]
            assert not any(x in result.lower() for x in ["execution failed", "command not found", "could not be downloaded", "traceback (most recent"]), result
            if expected:
                assert expected in result, result
            record("tool_passed", tool=plan[calls - 2][0])
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = await runtime._client_or_raise().get_sandbox(runtime.info.id)
            if state.get("state") == "parked":
                break
            await asyncio.sleep(1)
        else:
            raise RuntimeError("Harness failed to park")
        record("parked", call=calls)
        if calls <= len(plan):
            tool, arguments, _ = plan[calls - 1]
            assert tool in [t["function"]["name"] for t in body.get("tools", [])]
            message = {"role": "assistant", "content": None, "tool_calls": [{"id": f"probe-{calls}", "type": "function", "function": {"name": tool, "arguments": json.dumps(arguments)}}]}
        else:
            message = {"role": "assistant", "content": "OFFLINE_TOOLS_OK"}
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        finish = "tool_calls" if calls <= len(plan) else "stop"
        response = {"id": f"probe-{calls}", "object": "chat.completion", "created": int(time.time()),
                    "model": "parking-probe", "choices": [{"index": 0,"message": message,"finish_reason": finish}], "usage": usage}
        if not body.get("stream"):
            return web.json_response(response)
        delta = dict(message)
        if "tool_calls" in delta: delta["tool_calls"] = [dict(t,index=i) for i,t in enumerate(delta["tool_calls"])]
        chunks = [{**response, "object": "chat.completion.chunk", "choices": [{"index":0,"delta":delta,"finish_reason":None}]},
                  {**response, "object": "chat.completion.chunk", "choices": [{"index":0,"delta":{},"finish_reason":finish}]}]
        return web.Response(text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n",content_type="text/event-stream")


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
                "harness": {"id": args.harness},
                "runtime": runtime.config,
                "model": "parking-probe",
                "client": {
                    "type": "eval",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key_var": "PARKING_PROBE_UNUSED_KEY",
                },
                "max_turns": 8,
                "timeout": {"setup": 600, "rollout": 900},
            }
        )
        task = Task(
            TaskData(prompt="Run the requested shell command and report completion.", workdir="/tmp", network_allow=[])
        )
        async with Agent(config, interception=interception) as agent:
            trace = await agent.run(task, runtime=runtime)
        args.output.with_suffix(".trace.json").write_text(trace.model_dump_json(indent=2))
        assert trace.ok, str(trace.errors)
        assert calls == len(plan) + 1, calls
        assert (await runtime.read(filename)) == b"EDITED_SENTINEL"
        assert "sandbox_handle" not in trace.model_dump_json()
        record("passed", calls=calls, stop_condition=trace.stop_condition, tool_state_preserved=True)
    finally:
        await interception.stop()
        await runtime.stop()
        await runner.cleanup()
        record("cleanup_complete", id=name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harness", choices=["pi","prime-agent","opencode"], required=True)
    p.add_argument("--image", default="python:3.11-slim")
    p.add_argument("--image-type", choices=["name", "registry"], default="registry")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main(args))
