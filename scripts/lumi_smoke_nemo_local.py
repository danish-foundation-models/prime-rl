"""Exercise the real local runtime/interceptor/scorer with a scripted HTTP model."""

import argparse
import asyncio
import json
from pathlib import Path

import verifiers.v1 as vf
from aiohttp import web
from verifiers.v1.clients import EvalClientConfig, ModelContext
from verifiers.v1.utils.loaders import resolve_env_config


async def main(args):
    rows = {kind: json.loads((args.data / f"{kind}-train.jsonl").open().readline()) for kind in ["pivot", "workplace"]}
    rows["calendar"] = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Schedule one 30-minute event after 9am and before 11am."}]
        },
        "exp_cal_state": {"0": {"duration": 30, "constraint": "after 9am", "min_time": "09:00", "max_time": "11:00"}},
    }
    rows["instruction"] = {
        "responses_create_params": {"input": [{"role": "user", "content": "End with contest."}]},
        "instruction_id_list": ["last_word:last_word_answer"],
        "kwargs": [{"last_word": "contest"}],
        "prompt": "End with contest.",
    }

    async def completion(request):
        body = await request.json()
        kind = body["model"]
        message = {"role": "assistant", "content": None, "reasoning_content": "A short reasoning segment."}
        finished = False
        if kind == "pivot":
            action = rows[kind]["expected_action"]
            message["tool_calls"] = [
                {"id": "call_pivot", "type": "function", "function": {k: action[k] for k in ["name", "arguments"]}}
            ]
        elif kind == "workplace" and body["messages"][-1]["role"] != "tool":
            message["tool_calls"] = [
                {"id": f"call_{i}", "type": "function", "function": action}
                for i, action in enumerate(rows[kind]["ground_truth"])
            ]
        else:
            finished = True
            message["content"] = {
                "workplace": "Done.",
                "calendar": '[{"event_id":0,"start_time":"9am","duration":30}]',
                "instruction": "contest",
            }[kind]
        return web.json_response(
            {
                "id": "scripted",
                "object": "chat.completion",
                "created": 0,
                "model": kind,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop" if finished else "tool_calls"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        for kind, row in rows.items():
            path = args.output / f"{kind}-fixture.jsonl"
            path.write_text(json.dumps(row) + "\n")
            env = vf.load_environment(
                resolve_env_config(
                    {
                        "taskset": {"id": "nemo_local", "kind": kind, "dataset": str(path)},
                        "agent": {"harness": {"id": "nemo_local"}, "runtime": {"type": "subprocess"}, "max_turns": 4},
                        "interception": {"type": "server"},
                    }
                )
            )
            task = next(iter(env.taskset))
            ctx = ModelContext(
                model=kind,
                client=EvalClientConfig(base_url=f"http://127.0.0.1:{port}/v1", api_key_var="NEMO_SMOKE_UNUSED_KEY"),
            )
            async with env.serving():
                # Two simultaneous copies also exercise per-rollout simulator isolation.
                episodes = await asyncio.gather(env.run_episode(task, ctx), env.run_episode(task, ctx))
            for i, episode in enumerate(episodes):
                (args.output / f"{kind}-{i}-episode.json").write_text(episode.model_dump_json(indent=2))
                assert episode.ok, episode.model_dump_json(indent=2)[-5000:]
                trace = episode.traces[0]
                assert trace.info["nemo_local"]["task_reward"] == 1, trace.info
                assert len(trace.assistant_messages) == (2 if kind == "workplace" else 1)
                assert trace.assistant_messages[0].reasoning_content
            print(kind, "2 isolated full rollouts passed", flush=True)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(main(args))
