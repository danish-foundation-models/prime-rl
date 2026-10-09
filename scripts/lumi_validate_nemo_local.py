"""Audit every staged row and run adversarial reward checks without a model or sandbox."""

import argparse
import json
from collections import Counter
from pathlib import Path

from nemo_local._vendor.verifiable_instructions import instructions_registry
from nemo_local.grading import grade, instruction_metadata
from nemo_local.workplace import replay, same_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=["pivot", "instruction", "calendar", "workplace"],
        default=["pivot", "instruction", "calendar", "workplace"],
    )
    args = parser.parse_args()
    report = {}
    for kind in args.kinds:
        failures = []
        counts = Counter()
        baseline = replay([]) if kind == "workplace" else None
        for split in ["train", "validation"]:
            with (args.dataset_dir / f"{kind}-{split}.jsonl").open() as file:
                for index, line in enumerate(file):
                    row = json.loads(line)
                    counts["rows"] += 1
                    try:
                        if kind == "pivot":
                            gold = row["expected_action"]
                            assert grade(kind, row, "", [gold])[0] == 1
                            assert grade(kind, row, "", [gold, gold])[0] == 0
                            assert grade(kind, row, "Done.", [])[0] == 0
                        elif kind == "instruction":
                            vm = instruction_metadata(row)
                            assert vm["instruction_id_list"]
                            assert len(vm["instruction_id_list"]) == len(vm["kwargs"])
                            for ident, kwargs in zip(vm["instruction_id_list"], vm["kwargs"], strict=True):
                                validator = instructions_registry.INSTRUCTION_DICT[ident](ident)
                                validator.build_description(
                                    **{k: v for k, v in (kwargs or {}).items() if v is not None}
                                )
                                validator.check_following("This is a deliberately wrong answer.")
                                counts["checks"] += 1
                            assert grade(kind, row, "", [])[0] == 0
                        elif kind == "calendar":
                            assert row["exp_cal_state"]
                            assert grade(kind, row, "Done.", [])[0] == 0
                        else:
                            expected = replay(row["ground_truth"])
                            assert not same_state(expected, baseline), "no_op_ground_truth"
                            assert grade(kind, row, "Done.", row["ground_truth"])[0] == 1
                            assert grade(kind, row, "Done.", [])[0] == 0
                    except Exception as error:
                        failures.append(
                            {"split": split, "index": index, "error": f"{type(error).__name__}: {error}"[:500]}
                        )
        report[kind] = {"counts": dict(counts), "failures": failures}
        print(kind, dict(counts), "failures", len(failures), "examples", failures[:3], flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(value["failures"] for value in report.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
