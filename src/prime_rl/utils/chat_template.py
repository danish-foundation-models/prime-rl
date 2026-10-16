import json
from typing import Any


def ensure_tokenizer_pad_token(tokenizer: Any) -> None:
    """Preserve a configured pad ID, falling back to EOS only when absent."""
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")
    tokenizer.pad_token_id = eos_token_id


def normalize_messages(messages: Any, default_role: str) -> list[dict[str, Any]]:
    if messages is None:
        return []
    if isinstance(messages, str):
        return [{"role": default_role, "content": messages}]
    if isinstance(messages, dict):
        return [dict(messages)]
    if isinstance(messages, list):
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, str):
                normalized.append({"role": default_role, "content": message})
            elif isinstance(message, dict):
                normalized.append(dict(message))
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")
        return normalized
    raise TypeError(f"Unsupported messages container type: {type(messages)}")


def deserialize_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _deserialize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function", {})
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return {
            **tool_call,
            "function": {**function, "arguments": arguments},
        }

    deserialized_messages: list[dict[str, Any]] = []
    for message in messages:
        if "tool_calls" not in message:
            deserialized_messages.append(dict(message))
            continue

        tool_calls = message.get("tool_calls") or []
        deserialized_messages.append(
            {
                **message,
                "tool_calls": [_deserialize_tool_call(tc) for tc in tool_calls],
            }
        )

    return deserialized_messages
