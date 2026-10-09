"""Pin and stage public NeMo train data; never load HF during rollout startup."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from nemo_local.grading import grade_pivot
from nemo_local.workplace import replay, same_state


def quality_reason(kind, row):
    if kind == "pivot":
        expected = row["expected_action"]
        catalog = {t["name"]: t for t in row["responses_create_params"]["tools"]}
        if expected["name"] not in catalog:
            return "target_tool_not_advertised"
        if grade_pivot(row, "", [expected])[0] != 1:
            return "target_arguments_violate_schema"
    if kind == "workplace" and same_state(replay(row["ground_truth"]), replay([])):
        return "no_state_change_needs_semantic_grading"
    return None


SOURCES = {
    "pivot": (
        "Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        "cd26a12f54348e02a832d85f9da86621cd5dca4a",
        "train.jsonl",
    ),
    "instruction": (
        "Nemotron-RL-instruction_following",
        "3b253899665cb71334bb54c14eb5d91751beaad7",
        "instruction_following.jsonl",
    ),
    "calendar": (
        "Nemotron-RL-Instruction-Following-Calendar-v2",
        "556a3b1ab3eb12bab38327bf0ec4cdbeab452338",
        "train.jsonl",
    ),
    "workplace": ("Nemotron-RL-agent-workplace_assistant", "c86a908379e0a361a573c395e175d3c1aa128e6c", "train.jsonl"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def group_key(kind, row):
    if kind == "pivot":
        return digest(["pivot-trajectory", row["trajectory_id"]])
    messages = row["responses_create_params"]["input"]
    if kind == "calendar":
        # Keep all continuation variants of an initial calendar conversation together.
        return digest(next(m for m in messages if m.get("role") == "user"))
    return digest(messages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": {}, "holdout": "sha256(group key) modulo 20 == 0; Pivot grouped by trajectory_id"}
    for kind, (name, revision, filename) in SOURCES.items():
        repo = "nvidia/" + name
        path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
        rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
        counts = {"raw_train": len(rows), "excluded": {}, "train": 0, "validation": 0}
        seen = set()
        outputs = {"train": [], "validation": []}
        for row in rows:
            if kind == "pivot" and row["expected_action"]["type"] != "function_call":
                reason = "message_action_not_semantically_verifiable"
            elif kind == "calendar" and not row.get("exp_cal_state"):
                reason = "empty_calendar_unconditional_upstream_reward"
            elif digest(row) in seen:
                reason = "duplicate_row"
            else:
                reason = quality_reason(kind, row)
            if reason:
                counts["excluded"][reason] = counts["excluded"].get(reason, 0) + 1
                continue
            seen.add(digest(row))
            group = group_key(kind, row)
            split = "validation" if int(group[:16], 16) % 20 == 0 else "train"
            outputs[split].append(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
        files = {}
        for split, lines in outputs.items():
            output = args.output / f"{kind}-{split}.jsonl"
            content = "".join(lines).encode()
            if output.exists() and output.read_bytes() != content:
                raise FileExistsError(f"Refusing to overwrite different staged data: {output}")
            output.write_bytes(content)
            files[split] = {"path": str(output.resolve()), "sha256": hashlib.sha256(content).hexdigest()}
        manifest["sources"][kind] = {
            "repo": repo,
            "revision": revision,
            "filename": filename,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "counts": counts,
            "files": files,
        }
        print(kind, counts, flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
