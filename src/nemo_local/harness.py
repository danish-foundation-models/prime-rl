"""Local-only harness; policy requests still pass through verifiers interception."""

import json
import sys

import verifiers.v1 as vf
from verifiers.v1.dialects.chat import message_to_wire


class NeMoLocalHarnessConfig(vf.HarnessConfig):
    pass


class NeMoLocalHarness(vf.Harness[NeMoLocalHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MESSAGE_PROMPT = True
    SUPPORTS_MCP = False
    SUPPORTS_RESUME = False
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        if runtime.info.type != "subprocess":
            raise ValueError("nemo_local requires the local subprocess runtime")
        system, prompt = self.resolve_prompt(data)
        messages = [message_to_wire(m) for m in prompt]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        # Only public inputs reach the harness; targets stay with the scorer.
        payload = {
            "kind": data.kind,
            "messages": messages,
            "tools": data.row["responses_create_params"].get("tools", []),
        }
        path = "nemo-input.json"
        await runtime.write(path, json.dumps(payload).encode())
        return await runtime.run_program(
            [
                sys.executable,
                "-m",
                "nemo_local.program",
                f"--base-url={endpoint}",
                f"--api-key={secret}",
                f"--model={ctx.model}",
                f"--input={path}",
            ],
            self.config.resolved_env,
        )
