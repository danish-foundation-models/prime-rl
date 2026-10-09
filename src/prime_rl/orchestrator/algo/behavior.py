"""Conservative behavior flags over sampled assistant messages, not observations."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorFlags:
    malformed_tool_nodes: tuple[int, ...] = ()
    reasoning_only_end: bool = False

    @property
    def penalize(self) -> bool:
        return bool(self.malformed_tool_nodes) or self.reasoning_only_end


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part.get("text", "") for part in value if isinstance(part, dict))
    return ""


def _unquoted(text: str) -> str:
    text = re.sub(r"```.*?(?:```|$)|~~~.*?(?:~~~|$)", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    return re.sub(r"(?m)^\s*>.*$", "", text)


def detect_behavior(trace) -> BehaviorFlags:
    if trace.has_error:
        return BehaviorFlags()
    nodes = []
    malformed = []
    for index, node in enumerate(trace.nodes):
        if not node.sampled or not any(node.mask):
            continue
        message = node.message
        if not isinstance(message, dict):
            message = message.model_dump(exclude_none=True)
        if message.get("role") != "assistant":
            continue
        nodes.append((index, message))
        # Parsed calls have already passed the renderer. Never scan their argument
        # payloads, which may intentionally manipulate tool-format examples.
        if message.get("tool_calls"):
            continue
        text = _unquoted("\n".join(_text(message.get(k)) for k in ("reasoning_content", "content")))
        # Detect an unexecuted XML-shaped action regardless of what precedes the
        # function: any SPECIAL_n, another tag, plain text, or no opener at all.
        function = re.search(r"<function=[^>\n]+>", text)
        if function:
            body = text[function.end() :]
            if "<parameter=" in body and ("</function>" in body or "</tool_call>" in body):
                malformed.append(index)
    empty_end = False
    if nodes and not trace.is_truncated and trace.stop_condition in ("agent_completed", "finished"):
        index, message = nodes[-1]
        call = next((c for c in reversed(trace.calls) if c.node == index), None)
        empty_end = bool(
            call is not None
            and call.error is None
            and call.finish_reason == "stop"
            and _text(message.get("reasoning_content")).strip()
            and not _text(message.get("content")).strip()
            and not message.get("tool_calls")
        )
    return BehaviorFlags(tuple(malformed), empty_end)
