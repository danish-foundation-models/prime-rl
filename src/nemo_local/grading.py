"""Deterministic grading; invalid model answers score zero, grader faults propagate."""

import hashlib
import json
import math
import random
import re

import jsonschema

from ._vendor.calendar.utils import extract_json_list, grade_assistant_response, time_to_minutes


def strict_json(value):
    def pairs(items):
        result = {}
        for key, val in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = val
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid_constant)


def canonical(value):
    # Keep boolean distinct from integer (Python otherwise considers True == 1).
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def grade_pivot(row, text, calls):
    expected = row["expected_action"]
    if expected["type"] != "function_call":
        raise ValueError("Pivot message targets need semantic grading; not admitted here")
    if len(calls) != 1 or calls[0]["name"] != expected["name"]:
        return 0.0, {"reason": "wrong_call_count_or_name"}
    catalog = {t["name"]: t for t in row["responses_create_params"]["tools"]}
    try:
        args = strict_json(calls[0]["arguments"])
        jsonschema.validate(args, catalog[expected["name"]]["parameters"])
    except (ValueError, TypeError, jsonschema.ValidationError):
        return 0.0, {"reason": "invalid_arguments"}
    gold = strict_json(expected["arguments"])
    return float(canonical(args) == canonical(gold)), {"reason": "argument_comparison"}


def grade_calendar(row, text, calls):
    expected = row["exp_cal_state"]
    if not expected:
        raise ValueError("Empty calendars are excluded because upstream accepts arbitrary text")
    if calls or not text.strip():
        return 0.0, {"reason": "missing_final_calendar"}
    events = extract_json_list(text)
    try:
        if not events or len(events) != len(expected):
            return 0.0, {"reason": "event_count"}
        ids = [str(e["event_id"]) for e in events]
        if len(set(ids)) != len(ids) or set(ids) != set(expected):
            return 0.0, {"reason": "event_ids"}
        for event in events:
            duration = event["duration"]
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                return 0.0, {"reason": "invalid_duration"}
            time = event["start_time"].strip()
            if not re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d|(?:[1-9]|1[0-2])(?::[0-5]\d)?(?:am|pm)", time):
                return 0.0, {"reason": "invalid_time"}
            if time_to_minutes(time) + duration > 1440:
                return 0.0, {"reason": "past_midnight"}
    except (KeyError, TypeError, ValueError):
        return 0.0, {"reason": "malformed_calendar"}
    reward, reason = grade_assistant_response(text, expected)
    return float(reward), {"reason": reason}


def instruction_metadata(row):
    metadata = {key: row[key] for key in ["instruction_id_list", "prompt", "kwargs", "grading_mode"] if key in row}
    metadata.update(row.get("verifier_metadata") or {})
    metadata["kwargs"] = [dict(value or {}) for value in metadata["kwargs"]]
    for ident, kwargs in zip(metadata["instruction_id_list"], metadata["kwargs"], strict=True):
        if ident == "count:count_increment_word":
            for key in ["keyword1", "keyword2"]:
                value = kwargs.get(key)
                if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
                    kwargs[key] = value[0]
    return metadata


def grade_instruction(row, text, calls):
    from langdetect import DetectorFactory

    from ._vendor.verifiable_instructions import instructions_registry

    DetectorFactory.seed = 0
    random.seed(hashlib.sha256(canonical(row).encode()).hexdigest())
    metadata = instruction_metadata(row)
    ids, kwargs = metadata["instruction_id_list"], metadata["kwargs"]
    if not ids or len(ids) != len(kwargs):
        raise ValueError("Instruction IDs and kwargs must be nonempty and aligned")
    if calls or not text.strip():
        return 0.0, {"follow_instruction_list": [False] * len(ids)}
    passed = []
    for ident, arguments in zip(ids, kwargs, strict=True):
        instruction = instructions_registry.INSTRUCTION_DICT[ident](ident)
        instruction.build_description(**{k: v for k, v in (arguments or {}).items() if v is not None})
        passed.append(bool(instruction.check_following(text)))
    mode = metadata.get("grading_mode", "binary")
    if mode not in ["binary", "fraction"]:
        raise ValueError(f"Unknown grading mode: {mode}")
    reward = float(all(passed)) if mode == "binary" else sum(passed) / len(passed)
    return reward, {"follow_instruction_list": passed, "follow_all_instructions": all(passed)}


def grade_workplace(row, text, calls):
    from .workplace import expected_snapshot, replay, same_state

    gold = expected_snapshot(canonical(row["ground_truth"]))
    if not calls:
        return 0.0, {"reason": "no_actions"}
    advertised = {tool["name"] for tool in row["responses_create_params"].get("tools", [])}
    if any(call["name"] not in advertised for call in calls):
        return 0.0, {"reason": "unadvertised_action"}
    try:
        # Unlike upstream, exceptions cannot silently disappear during replay.
        actual = replay(calls)
    except (ValueError, TypeError, KeyError, jsonschema.ValidationError):
        return 0.0, {"reason": "invalid_action"}
    return float(same_state(actual, gold)), {"reason": "final_state_comparison"}


GRADERS = {
    "pivot": grade_pivot,
    "calendar": grade_calendar,
    "instruction": grade_instruction,
    "workplace": grade_workplace,
}


def grade(kind, row, text, calls):
    value, details = GRADERS[kind](row, text, calls)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid reward: {value}")
    return value, details
