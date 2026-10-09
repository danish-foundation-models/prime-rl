import copy

import pytest

from nemo_local.grading import grade
from nemo_local.workplace import execute, new_session, same_state, snapshot


def pivot_row():
    return {
        "expected_action": {"type": "function_call", "name": "lookup", "arguments": '{"id":1}'},
        "responses_create_params": {
            "tools": [
                {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                }
            ]
        },
    }


def test_pivot_rejects_extra_calls_and_malformed_arguments():
    row = pivot_row()
    correct = row["expected_action"]
    assert grade("pivot", row, "", [correct])[0] == 1
    assert grade("pivot", row, "Done", [])[0] == 0
    assert grade("pivot", row, "", [correct, correct])[0] == 0
    for arguments in ['{"id":true}', '{"id":1,"id":2}', '{"id":NaN}', "[]"]:
        assert grade("pivot", row, "", [{**correct, "arguments": arguments}])[0] == 0


def test_calendar_ids_and_final_answer():
    row = {
        "exp_cal_state": {"0": {"duration": 30, "constraint": "after 9am", "min_time": "09:00", "max_time": "11:00"}}
    }
    good = '[{"event_id":0,"start_time":"9am","duration":30}]'
    assert grade("calendar", row, good, [])[0] == 1
    assert grade("calendar", row, "Thinking about it...", [])[0] == 0
    assert grade("calendar", row, good.replace("9am", "8am"), [])[0] == 0
    assert grade("calendar", row, good[:-1] + "," + good[1:], [])[0] == 0
    assert grade("calendar", row, good.replace("9am", "25:00"), [])[0] == 0


def test_instruction_published_singleton_lists_and_empty_response():
    row = {
        "instruction_id_list": ["count:count_increment_word"],
        "kwargs": [{"keyword1": ["help"], "keyword2": ["dump"]}],
        "prompt": "Use help once and dump twice.",
    }
    original = copy.deepcopy(row)
    assert grade("instruction", row, "help dump dump", [])[0] == 1
    assert grade("instruction", row, "help dump", [])[0] == 0
    assert grade("instruction", row, "", [])[0] == 0
    assert row == original


def test_workplace_fresh_state_and_unknown_tool():
    left, right = new_session(), new_session()
    baseline = snapshot(right)
    execute(left, "email_send_email", {"recipient": "test@example.org", "subject": "Simulation", "body": "Hello"})
    assert same_state(snapshot(right), baseline)
    assert not same_state(snapshot(left), baseline)
    with pytest.raises(ValueError, match="Unknown"):
        execute(left, "__import__", {"name": "os"})


def test_workplace_replay_matches_runtime_rejections():
    from nemo_local.workplace import schemas

    action = {
        "name": "email_send_email",
        "arguments": '{"recipient":"test@example.org","subject":"Test","body":"Hello"}',
    }
    row = {"ground_truth": [action], "responses_create_params": {"tools": list(schemas(new_session()).values())}}
    assert grade("workplace", row, "Done", [action])[0] == 1
    malformed = {**action, "arguments": action["arguments"].replace('"body":"Hello"', '"body":"Wrong","body":"Hello"')}
    assert grade("workplace", row, "Done", [malformed])[0] == 0
    row["responses_create_params"]["tools"] = []
    assert grade("workplace", row, "Done", [action])[0] == 0
