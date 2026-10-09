"""A chat-completions loop with only fixed Workplace simulator functions."""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI


def chat_tools(tools):
    return [
        {
            "type": "function",
            "function": {key: value for key, value in tool.items() if key in ["name", "description", "parameters"]},
        }
        for tool in tools
    ]


async def run(args):
    payload = json.loads(Path(args.input).read_text())
    tools = chat_tools(payload["tools"])
    messages = payload["messages"]
    workplace = payload["kind"] == "workplace"
    if workplace:
        import jsonschema

        from .grading import strict_json
        from .workplace import execute, new_session

        session = new_session()
    async with AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=2, timeout=600) as client:
        while True:
            response = await client.chat.completions.create(model=args.model, messages=messages, tools=tools or None)
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            # Pivot grades the sampled call directly, without executing any API.
            if not workplace or not message.tool_calls:
                return
            for call in message.tool_calls:
                try:
                    if call.function.name not in {t["name"] for t in payload["tools"]}:
                        raise ValueError("Tool is not advertised for this task")
                    result = execute(session, call.function.name, strict_json(call.function.arguments))
                    content = json.dumps(result, ensure_ascii=False, default=str)
                except (ValueError, TypeError, KeyError, jsonschema.ValidationError) as error:
                    content = json.dumps({"error": str(error)[:1000]})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})


def main():
    parser = argparse.ArgumentParser()
    for name in ["base-url", "api-key", "model", "input"]:
        parser.add_argument("--" + name, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
