from copy import deepcopy
from types import SimpleNamespace

from prime_rl.orchestrator.loop_annotations import analyze_nodes, annotate_episode


def conversation(pairs):
    nodes = []
    for i, (command, output) in enumerate(pairs):
        nodes.append(
            {
                "parent": len(nodes) - 1 if nodes else None,
                "sampled": True,
                "mask": [True] * 10,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": str(i), "name": "bash", "arguments": {"command": command}}],
                },
            }
        )
        nodes.append(
            {
                "parent": len(nodes) - 1,
                "sampled": False,
                "message": {"role": "tool", "tool_call_id": str(i), "content": output},
            }
        )
    return nodes


def test_unchanged_cycles_and_changed_results():
    for pairs, expected in [
        ([("pytest", "failed")] * 3, True),
        ([("ls", "a"), ("cat a", "text")] * 3, True),
        ([("pytest", "failed"), ("pytest", "passed"), ("pytest", "passed")], False),
        ([("pytest", "failed"), ("edit file", ""), ("pytest", "failed")], False),
        ([("pytest", "failed")] * 2, False),
    ]:
        nodes = conversation(pairs)
        old = deepcopy(nodes)
        metrics, info = analyze_nodes(nodes)
        assert info["suspected"] is expected
        assert metrics["loop/suspected"] == float(expected)
        assert nodes == old


def test_branches_and_prompt_copies_do_not_create_cycles():
    nodes = conversation([("pytest", "failed")] * 3)
    nodes[2]["parent"] = None
    nodes[4]["parent"] = None
    assert not analyze_nodes(nodes)[1]["suspected"]
    nodes = conversation([("pytest", "failed")] * 3)
    for node in nodes:
        node["sampled"] = False
    assert not analyze_nodes(nodes)[1]["suspected"]


def test_reasoning_repetition_locations_and_short_normal_text():
    sentence = " ".join(f"word{i}" for i in range(32))
    nodes = [
        {
            "sampled": True,
            "parent": None,
            "mask": [True] * 120,
            "message": {"role": "assistant", "reasoning_content": " ".join([sentence] * 4)},
        }
    ]
    metrics, info = analyze_nodes(nodes)
    assert info["events"][0]["kind"] == "reasoning_repetition"
    assert metrics["loop/reasoning_repeated_words"] >= 64
    assert 0 < metrics["loop/reasoning_repeated_fraction"] <= 1
    assert metrics["loop/flagged_output_tokens"] == 120
    nodes[0]["message"]["reasoning_content"] = "I should rerun the tests after this edit."
    assert not analyze_nodes(nodes)[1]["suspected"]


def test_tool_ids_pair_outputs_and_argument_json_is_canonical():
    nodes = conversation([("pytest", "failed")] * 3)
    nodes[2]["message"]["tool_calls"][0]["arguments"] = '{ "command" : "pytest" }'
    assert analyze_nodes(nodes)[1]["suspected"]
    nodes[-1]["message"]["tool_call_id"] = "unknown"
    assert not analyze_nodes(nodes)[1]["suspected"]


def test_annotation_only_changes_diagnostics():
    class Message:
        def __init__(self, value):
            self.value = value

        def model_dump(self, **kwargs):
            return self.value

    nodes = [
        SimpleNamespace(
            parent=n.get("parent"), sampled=n["sampled"], mask=n.get("mask", []), message=Message(n["message"])
        )
        for n in conversation([("pytest", "failed")] * 3)
    ]
    trace = SimpleNamespace(
        nodes=nodes,
        rewards={"solved": 1},
        metrics={"existing": 2},
        info={"existing": True},
        ok=True,
        stop_condition="agent_completed",
        errors=[],
        advantages=[1, -1],
    )
    before = deepcopy(trace.__dict__)
    annotate_episode(SimpleNamespace(traces=[trace]))
    for field in ("rewards", "ok", "stop_condition", "errors", "advantages"):
        assert getattr(trace, field) == before[field]
    assert trace.metrics["existing"] == 2
    assert trace.info["existing"] is True
    assert trace.info["loop_annotations"]["observation_only"]


def test_short_repeated_quotation_is_evidence_but_not_a_loop_flag():
    quote = " ".join(f"quote{i}" for i in range(16))
    text = " ".join(f"distinct{i}" for i in range(300)) + " " + " ".join([quote] * 3)
    metrics, info = analyze_nodes(
        [{"parent": None, "sampled": True, "message": {"role": "assistant", "reasoning_content": text}}]
    )
    assert metrics["loop/reasoning_repeated_words"] > 0
    assert not info["suspected"]
    assert info["events"][0]["substantial"] is False


def test_repetition_that_recovers_does_not_get_loop_flag():
    nodes = conversation([("pytest", "failed")] * 3 + [("edit file", "changed")])
    metrics, info = analyze_nodes(nodes)
    assert not info["suspected"]
    assert metrics["loop/nonterminal_repetition_events"] > 0
    reasoning = " ".join([" ".join(f"word{i}" for i in range(32))] * 4)
    nodes = [{"sampled": True, "message": {"role": "assistant", "reasoning_content": reasoning}}]
    assert analyze_nodes(nodes)[1]["suspected"]
    nodes[0]["message"]["reasoning_content"] += " " + " ".join(f"new{i}" for i in range(32))
    assert not analyze_nodes(nodes)[1]["suspected"]
    nodes[0]["message"]["reasoning_content"] = reasoning
    nodes.append({"parent": 0, "sampled": True, "message": {"role": "assistant", "content": "Solved."}})
    assert not analyze_nodes(nodes)[1]["suspected"]


def test_content_and_reasoning_loops_with_small_variations():
    phrase = "I need to inspect the implementation again because the same issue still remains unresolved here"
    text = " ".join(f"{phrase} attempt{i}." for i in range(12))
    for field in ("content", "reasoning_content"):
        nodes = [{"sampled": True, "message": {"role": "assistant", field: text}}]
        metrics, info = analyze_nodes(nodes)
        assert info["suspected"]
        assert metrics["loop/text_tail_candidates"] == 1
        nodes[0]["message"][field] += " " + " ".join(f"newword{i}" for i in range(150))
        assert not analyze_nodes(nodes)[1]["suspected"]
    nodes = [{"sampled": True, "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}]
    assert analyze_nodes(nodes)[1]["suspected"]


def test_action_loop_despite_different_outputs_and_pending_last_call():
    nodes = conversation([("pytest", f"failed in {i} seconds") for i in range(5)])
    assert analyze_nodes(nodes)[1]["suspected"]
    assert analyze_nodes(nodes[:-1])[1]["suspected"]
    assert any(e["kind"] == "repeated_action_cycle" for e in analyze_nodes(nodes)[1]["events"])
    progressing = conversation([("pytest", "failed"), ("edit file", "changed"), ("pytest", "passed")])
    assert not analyze_nodes(progressing)[1]["suspected"]
    for n in nodes:
        if n["sampled"]:
            n["parent"] = None
    assert not analyze_nodes(nodes)[1]["suspected"]


def test_reviewed_real_traces_distinguish_analysis_loops_from_code_repetition():
    import json
    from pathlib import Path

    fixtures = json.loads((Path(__file__).parent / "fixtures/loop_review_r9.json").read_text())
    for fixture in fixtures:
        nodes = [{"sampled": True, "message": fixture["message"]}]
        assert analyze_nodes(nodes)[1]["suspected"] == fixture["expected"], fixture["episode_id"]
