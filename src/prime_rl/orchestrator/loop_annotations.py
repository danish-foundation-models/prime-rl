"""Observation-only repetition heuristics over sampled messages, never reward shaping."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

VERSION = 4
SHINGLE_WORDS = 16
MIN_REPEATS = 3
MAX_CYCLE = 4
MAX_EVENTS = 20
MAX_WORDS = 131072
MIN_REPEATED_WORDS = 64
MIN_REPEATED_FRACTION = 0.25


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _call_key(call: dict) -> str:
    function = call.get("function") or call
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return _digest([function.get("name"), arguments])


def _text_repeats(text: str, budget: int) -> tuple[list[tuple[int, int]], int, bool]:
    words = []
    capped = False
    for match in re.finditer(r"\S+", text):
        if len(words) == budget:
            capped = True
            break
        words.append(match.group())
    positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
    intervals = []
    for index in range(len(words) - SHINGLE_WORDS + 1):
        key = tuple(words[index : index + SHINGLE_WORDS])
        occurrences = positions[key]
        if occurrences and index - occurrences[-1] < SHINGLE_WORDS:
            continue
        occurrences.append(index)
        if len(occurrences) == MIN_REPEATS:
            intervals.extend((i, i + SHINGLE_WORDS) for i in occurrences[1:])
        elif len(occurrences) > MIN_REPEATS:
            intervals.append((index, index + SHINGLE_WORDS))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged, len(words), capped


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            part.get("text", "") for part in value if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _tail_repetition(text: str) -> dict:
    # Scan the suffix, not the prefix: long generations must not hide their loop.
    # Quoted code and edit payloads often repeat intentionally.
    prose = re.sub(r"```.*?(?:```|$)", " ", text, flags=re.S)
    prose = re.sub(r"<function=.*?(?:</function>|$)", " ", prose, flags=re.S)
    prose = re.sub(r"(?m)^>.*$", " ", prose)
    words = re.findall(r"\w+", prose.casefold())
    offset = max(0, len(words) - 8192)
    words = words[offset:]

    def coverage(width: int, previous_required: int) -> list[bool]:
        covered = [False] * len(words)
        occurrences = defaultdict(list)
        for i in range(len(words) - width + 1):
            key = tuple(words[i : i + width])
            previous = occurrences[key]
            if previous and i - previous[-1] < width:
                continue
            if len(previous) >= previous_required:
                covered[i : i + width] = [True] * width
            previous.append(i)
        return covered

    def fraction(values: list[bool], size: int) -> float:
        tail = values[-size:]
        return sum(tail) / len(tail) if tail else 0.0

    covered = coverage(5, 2)
    dense = sum(covered) >= 64 and fraction(covered, 128) >= 0.5 and fraction(covered, 32) >= 0.5
    # Sustained low novelty catches revisiting an argument with varying wording.
    # It is weaker evidence than a literal cycle and is labelled separately.
    trigrams = coverage(3, 1)
    bigrams = coverage(2, 1)
    low_novelty = (
        len(words) >= 768
        and fraction(trigrams, 512) >= 0.45
        and fraction(trigrams, 128) >= 0.35
        and fraction(bigrams, 512) >= 0.6
    )
    return {
        "repeated_words": sum(covered),
        "tail_repeated_fraction": fraction(covered, 128),
        "last_32_repeated_fraction": fraction(covered, 32),
        "word_offset": offset,
        "words_scanned": len(words),
        "scan_incomplete": offset > 0,
        "prose_trigram_fraction_512": fraction(trigrams, 512),
        "prose_trigram_fraction_128": fraction(trigrams, 128),
        "prose_bigram_fraction_512": fraction(bigrams, 512),
        "mode": "literal_repetition" if dense else "low_novelty",
        "substantial": dense or low_novelty,
    }


def _additional_candidates(nodes: list[dict]) -> list[dict]:
    events = []
    histories = []
    last_assistant = max(
        (i for i, n in enumerate(nodes) if n.get("sampled") and (n.get("message") or {}).get("role") == "assistant"),
        default=-1,
    )
    for index, node in enumerate(nodes):
        parent = node.get("parent")
        history = histories[parent] if isinstance(parent, int) and 0 <= parent < index else ()
        message = node.get("message") or {}
        if node.get("sampled") and message.get("role") == "assistant":
            fields = [(field, _text_value(message.get(field))) for field in ("reasoning_content", "content")]
            fields = [(field, text) for field, text in fields if text]
            if fields:
                # Content follows reasoning; evaluate their joint ending, rather than
                # assuming any content either proves recovery or must be ignored.
                observation = _tail_repetition("\n".join(text for _, text in fields))
                if observation["substantial"]:
                    events.append(
                        {
                            "kind": "text_repetition",
                            "field": " + ".join(field for field, _ in fields),
                            "node": index,
                            "persistent_to_trace_end": index == last_assistant,
                            **observation,
                        }
                    )
            # Count turns, not parallel identical calls within one response.
            calls = message.get("tool_calls") or []
            if calls:
                history = (*history, (_digest([_call_key(call) for call in calls]), index))[-32:]
                for period in range(1, 5):
                    if len(history) < 4 * period:
                        continue
                    tail = history[-4 * period :]
                    if all(item[0] == tail[i % period][0] for i, item in enumerate(tail)):
                        events.append(
                            {
                                "kind": "repeated_action_cycle",
                                "period": period,
                                "repetitions": 4,
                                "assistant_nodes": [item[1] for item in tail],
                                "persistent_to_trace_end": index == last_assistant,
                                "outputs_required_identical": False,
                            }
                        )
                        break
        histories.append(history)
    return events


def analyze_nodes(nodes: list[dict]) -> tuple[dict[str, float], dict]:
    """Analyze each sampled node once. Tool history follows physical parent links.

    Exact outputs are required: normalization must not hide changed test results.
    Prefix nodes copied into prompts are not model actions and are not counted.
    Text repetition is measured in whitespace words, not tokenizer tokens.
    """
    states: list[tuple[dict, tuple]] = []
    events = []
    flagged_nodes: set[int] = set()
    repeated_tools: set[int] = set()
    event_count = reasoning_words = repeated_words = tool_observations = 0
    incomplete = False
    recovered = persistent = 0
    last_activity = max(
        (i for i, n in enumerate(nodes) if n.get("sampled") or (n.get("message") or {}).get("role") == "tool"),
        default=-1,
    )
    for index, node in enumerate(nodes):
        parent = node.get("parent")
        pending, history = states[parent] if isinstance(parent, int) and 0 <= parent < index else ({}, ())
        pending = dict(pending)
        message = node.get("message") or {}
        role = message.get("role")
        if role == "assistant" and node.get("sampled"):
            for call in message.get("tool_calls") or []:
                if call.get("id"):
                    pending[call["id"]] = (_call_key(call), index)
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                spans, count, capped = _text_repeats(reasoning, max(0, MAX_WORDS - reasoning_words))
                reasoning_words += count
                incomplete |= capped
                if spans:
                    node_repeated = sum(end - start for start, end in spans)
                    repeated_words += node_repeated
                    substantial = node_repeated >= MIN_REPEATED_WORDS and node_repeated / count >= MIN_REPEATED_FRACTION
                    at_tail = False  # Prefix shingles are evidence; suffix checks decide flags.
                    if at_tail:
                        flagged_nodes.add(index)
                        persistent += 1
                    elif substantial:
                        recovered += 1
                    event_count += 1
                    if at_tail and len(events) >= MAX_EVENTS:
                        events.pop()
                    if len(events) < MAX_EVENTS:
                        events.append(
                            {
                                "kind": "reasoning_repetition",
                                "node": index,
                                "word_spans": spans[:MAX_EVENTS],
                                "span_count": len(spans),
                                "repeated_words": node_repeated,
                                "substantial": substantial,
                                "persistent_to_trace_end": at_tail,
                            }
                        )
        elif role == "tool" and message.get("tool_call_id") in pending and message.get("content") is not None:
            key, assistant = pending.pop(message["tool_call_id"])
            history = (*history, (key, _digest(message.get("content")), assistant, index))[-MAX_CYCLE * MIN_REPEATS :]
            tool_observations += 1
            for period in range(1, MAX_CYCLE + 1):
                if len(history) < period * MIN_REPEATS:
                    continue
                tail = history[-period * MIN_REPEATS :]
                signatures = [item[:2] for item in tail]
                if any(signatures[i] != signatures[i % period] for i in range(period, len(signatures))):
                    continue
                repeated_tools.update(item[3] for item in tail[period:])
                at_tail = index == last_activity
                if at_tail:
                    flagged_nodes.update(item[2] for item in tail[period:])
                    persistent += 1
                else:
                    recovered += 1
                event_count += 1
                if at_tail and len(events) >= MAX_EVENTS:
                    events.pop()
                if len(events) < MAX_EVENTS:
                    events.append(
                        {
                            "kind": "unchanged_tool_cycle",
                            "persistent_to_trace_end": at_tail,
                            "period": period,
                            "repetitions": MIN_REPEATS,
                            "assistant_nodes": [item[2] for item in tail],
                            "tool_nodes": [item[3] for item in tail],
                        }
                    )
                break
        elif role == "tool":
            # An unpaired/unknown result may represent intervening activity.
            history = ()
        states.append((pending, history))
    additional = _additional_candidates(nodes)
    for event in additional:
        event_count += 1
        if event["persistent_to_trace_end"]:
            flagged_nodes.update(event.get("assistant_nodes", [event.get("node")]))
            persistent += 1
            if len(events) >= MAX_EVENTS:
                events.pop()
        else:
            recovered += 1
        if len(events) < MAX_EVENTS:
            events.append(event)
    output_tokens = sum(sum(nodes[i].get("mask") or []) for i in flagged_nodes)
    metrics = {
        "loop/suspected": float(bool(flagged_nodes)),
        "loop/events": float(event_count),
        "loop/text_tail_candidates": float(
            sum(e["kind"] == "text_repetition" and e["persistent_to_trace_end"] for e in additional)
        ),
        "loop/action_tail_candidates": float(
            sum(e["kind"] == "repeated_action_cycle" and e["persistent_to_trace_end"] for e in additional)
        ),
        "loop/persistent_tail_events": float(persistent),
        "loop/nonterminal_repetition_events": float(recovered),
        "loop/reasoning_words_scanned": float(reasoning_words),
        "loop/reasoning_repeated_words": float(repeated_words),
        "loop/reasoning_repeated_fraction": repeated_words / reasoning_words if reasoning_words else 0.0,
        "loop/tool_observations": float(tool_observations),
        "loop/repeated_tool_observations": float(len(repeated_tools)),
        "loop/flagged_output_tokens": float(output_tokens),
        "loop/scan_incomplete": float(incomplete),
    }
    info = {
        "version": VERSION,
        "observation_only": True,
        "suspected": bool(flagged_nodes),
        "flag_requires": "repetition_persists_to_last_observed_activity",
        "reasoning_scope": "within_sampled_message",
        "text_fields": ["reasoning_content", "content"],
        "text_tail_window_words": 128,
        "action_only_min_repeats": 4,
        "tool_scope": "physical_branch",
        "shingle_words": SHINGLE_WORDS,
        "min_repeats": MIN_REPEATS,
        "min_repeated_words": MIN_REPEATED_WORDS,
        "min_repeated_fraction": MIN_REPEATED_FRACTION,
        "max_cycle": MAX_CYCLE,
        "flagged_nodes": sorted(flagged_nodes),
        "events": events,
        "events_omitted": max(0, event_count - len(events)),
        "scan_incomplete": incomplete,
    }
    return metrics, info


def annotate_episode(episode: Any) -> None:
    """Only append diagnostic metadata; do not alter any training/scoring fields."""
    for trace in episode.traces:
        nodes = [
            {
                "parent": node.parent,
                "sampled": node.sampled,
                "mask": node.mask,
                "message": node.message.model_dump(mode="json", exclude_none=True),
            }
            for node in trace.nodes
        ]
        metrics, info = analyze_nodes(nodes)
        info["ending"] = {
            "stop_condition": trace.stop_condition,
            "has_error": bool(getattr(trace, "has_error", not trace.ok)),
            "truncated": bool(getattr(trace, "is_truncated", False)),
        }
        trace.metrics.update(metrics)
        trace.info["loop_annotations"] = info
