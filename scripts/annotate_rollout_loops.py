"""Inspect recorded rollouts; --write adds a separate diagnostic annotation stream."""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from prime_rl.monitors.file.traces import get_annotations_dir, get_index_path, get_trace_stream
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl, open_chunk
from prime_rl.monitors.file.traces.update import make_update, update_index_row
from prime_rl.orchestrator.loop_annotations import VERSION, analyze_nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stream_dir = get_trace_stream(args.run)
    annotation_dir = get_annotations_dir(args.run) / f"loops-v{VERSION}"
    index = stream = source = None
    if args.write:
        # Exclusive creation avoids competing writers or overwriting earlier evidence.
        annotation_dir.mkdir(parents=True, exist_ok=False)
        index = get_index_path(annotation_dir).open("xb")
        stream = ChunkedJsonl(annotation_dir, 64 * 1024**2, compress=False)
    counts = defaultdict(Counter)
    examples = []
    chunk_number = None
    started = time.perf_counter()
    try:
        with get_index_path(stream_dir).open() as arrivals:
            for line in arrivals:
                row = json.loads(line)
                chunk = row.get("chunk", 0)
                if chunk != chunk_number:
                    if source is not None:
                        source.close()
                    source = open_chunk(stream_dir, chunk)
                    chunk_number = chunk
                source.seek(row["offset"])
                episode = json.loads(source.readline())
                for trace in episode.get("traces", []):
                    metrics, info = analyze_nodes(trace.get("nodes", []))
                    info["ending"] = {
                        "stop_condition": trace.get("stop_condition"),
                        "has_error": not trace.get("ok", False),
                        "last_finish_reason": (trace.get("calls") or [{}])[-1].get("finish_reason"),
                    }
                    env = row.get("env", "unknown")
                    counts[env]["traces"] += 1
                    counts[env]["flagged"] += bool(info["suspected"])
                    counts[env]["errors"] += not trace.get("ok", False)
                    if info["suspected"] and len(examples) < 20:
                        examples.append(
                            {
                                "episode": episode["id"],
                                "trace": trace["id"],
                                "env": env,
                                "ok": trace.get("ok"),
                                "observations": info,
                            }
                        )
                    if stream is not None:
                        update = make_update(trace["id"], info={"loop_annotations": info, "loop_metrics": metrics})
                        written_chunk, offset = stream.append((json.dumps(update) + "\n").encode())
                        stream.flush()
                        index.write((json.dumps(update_index_row(update, written_chunk, offset)) + "\n").encode())
                        index.flush()
        report = {
            "version": VERSION,
            "observation_only": True,
            "seconds": time.perf_counter() - started,
            "sources": dict(counts),
            "examples": examples,
        }
        if args.write:
            annotation_dir.with_name(annotation_dir.name + "-summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    finally:
        if source is not None:
            source.close()
        if stream is not None:
            stream.close()
        if index is not None:
            index.close()


if __name__ == "__main__":
    main()
