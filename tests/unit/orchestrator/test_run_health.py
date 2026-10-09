"""Operational stop decisions use error records and useful-work counters."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def watcher():
    path = Path(__file__).resolve().parents[3] / "scripts/lumi_watch_rl_pilot.py"
    spec = importlib.util.spec_from_file_location("run_health_watch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("error,expected", [
    ({"type": "ImageBuildFailure", "message": "build failed"}, "unavailable_task"),
    ({"type": "SandboxError", "message": "ucloud sandbox provisioning failed: Cached image build failure for task: apt failed"}, "unavailable_task"),
    ({"type": "SandboxError", "message": "ucloud sandbox provisioning failed: request timed out"}, "other_errors"),
    ({"type": "HarnessError", "message": "managed_process_read_unavailable"}, "other_errors"),
    ({"type": "SandboxNodeLost", "message": "node_lost"}, "node_lost"),
])
def test_image_errors_are_distinct_from_backend_errors(watcher, tmp_path, error, expected):
    path = tmp_path / "monitors/file/traces/stream/00000.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"traces": [{"ok": False, "errors": [error],
        "generated_text": "ImageBuildFailure: invented model output"}]}))
    assert watcher.trace_failure_kind(tmp_path, {"chunk": 0, "offset": 0}) == expected


def test_progress_watch_does_not_stop_long_active_tasks(watcher):
    health = watcher.ProgressWatch(0)
    health.observe(10, successful_episodes=0, trainer_step=0,
                   metrics=[{"inference/server0.0/generation_tokens_total": 100}])
    assert health.failure(609, idle_seconds=600, startup_seconds=3600) is None
    health.observe(609, successful_episodes=0, trainer_step=0,
                   metrics=[{"inference/server0.0/generation_tokens_total": 101}])
    assert health.failure(1208, idle_seconds=600, startup_seconds=3600) is None
    assert health.failure(1209, idle_seconds=600, startup_seconds=3600) is not None
    # A repeated scrape or empty timestamp update is not useful work.
    health.observe(1210, successful_episodes=0, trainer_step=0,
                   metrics=[{"time": 1210, "inference/server0.0/generation_tokens_total": 101}])
    assert health.failure(1210, idle_seconds=600, startup_seconds=3600) is not None
    health.observe(1211, successful_episodes=1, trainer_step=0)
    assert health.failure(1211, idle_seconds=600, startup_seconds=3600) is None


def test_startup_has_a_separate_budget(watcher):
    health = watcher.ProgressWatch(0)
    health.observe(100, successful_episodes=0, trainer_step=0,
                   metrics=[{"inference/server0.0/generation_tokens_total": 0}])
    assert health.failure(3599, idle_seconds=600, startup_seconds=3600) is None
    assert health.failure(3600, idle_seconds=600, startup_seconds=3600) is not None


def test_metric_reader_does_not_consume_partial_rows(watcher, tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"time":1}\n{"time":')
    rows, offset = watcher.read_metric_rows(path, 0)
    assert rows == [{"time": 1}]
    with path.open("ab") as handle:
        handle.write(b'2}\n')
    rows, offset = watcher.read_metric_rows(path, offset)
    assert rows == [{"time": 2}]
    assert offset == path.stat().st_size


def test_metric_reader_ignores_previous_attempt(watcher, tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"time":1,"inference/server0.0/generation_tokens_total":10000}\n'
                    '{"time":10,"inference/server0.0/generation_tokens_total":0}\n')
    rows, offset = watcher.read_metric_rows(path, 0, after_time=10)
    assert rows == [{"time": 10, "inference/server0.0/generation_tokens_total": 0}]
    assert offset == path.stat().st_size
    health = watcher.ProgressWatch(10)
    health.observe(10, successful_episodes=0, trainer_step=0, metrics=rows)
    assert not health.active


def test_low_throughput_stops_even_with_occasional_progress(watcher):
    health = watcher.ProgressWatch(0, min_generation_tps=100, throughput_window=60, throughput_warmup=120)
    for now in [0, 30, 60, 90, 119]:
        health.observe(now, successful_episodes=now, trainer_step=0,
                       metrics=[{"inference/server0.0/generation_tokens_total": 1 + now * 10}])
        assert health.failure(now, idle_seconds=600, startup_seconds=3600) is None
    health.observe(120, successful_episodes=120, trainer_step=0,
                   metrics=[{"inference/server0.0/generation_tokens_total": 1201}])
    assert health.generation_tps == 10
    assert "low generation throughput" in health.failure(120, idle_seconds=600, startup_seconds=3600)
    assert health.failure(120, idle_seconds=600, startup_seconds=3600, training_complete=True) is None
    assert "no token" in health.failure(721, idle_seconds=600, startup_seconds=3600, training_complete=True)


def test_weight_pause_and_counter_reset_start_new_throughput_windows(watcher):
    health = watcher.ProgressWatch(0, min_generation_tps=100, throughput_window=60, throughput_warmup=60)
    health.observe(0, successful_episodes=0, trainer_step=0,
                   metrics=[{"inference/server0.0/generation_tokens_total": 100}])
    health.observe(60, successful_episodes=0, trainer_step=0,
                   metrics=[{"inference/server0.0/generation_tokens_total": 60100}])
    assert health.generation_tps == 1000
    assert health.failure(60, idle_seconds=600, startup_seconds=3600) is None
    health.observe(120, successful_episodes=0, trainer_step=0,
                   metrics=[{"dispatcher/policy_update_pending": 1}])
    assert health.generation_tps is None
    health.observe(180, successful_episodes=0, trainer_step=1,
                   metrics=[{"dispatcher/policy_update_pending": 0}])
    assert health.generation_tps is None
    health.observe(210, successful_episodes=0, trainer_step=1,
                   metrics=[{"inference/server0.0/generation_tokens_total": 10}])
    assert health.generation_tps is None
    assert health.failure(210, idle_seconds=600, startup_seconds=3600) is None


def test_monitor_deduplicates_shared_build_incidents_only(watcher):
    from collections import deque
    recent = deque(maxlen=50)
    for i in range(34):
        row = {"id": str(i)}
        recent.append((True, watcher.failure_incident([], row)))
    for incident in ("a" * 32, "b" * 32):
        error = {"type": "ImageBuildPollingError", "message": f"Image build status unavailable [incident={incident}] build=example"}
        assert watcher.classify_trace_errors([error]) == "other_errors"
        for i in range(8):
            recent.append((False, watcher.failure_incident([error], {"id": f"{incident}-{i}"})))
    assert watcher.failure_window(recent) == {"eligible_episodes": 50, "failed_episodes": 16, "failure_incidents": 2}
    # Ordinary sandbox failures remain independent even with identical messages.
    recent.clear()
    for i in range(50):
        errors = [{"type": "SandboxError", "message": "read failed"}]
        recent.append((i >= 16, watcher.failure_incident(errors, {"id": str(i)})))
    assert watcher.failure_window(recent)["failure_incidents"] == 16
    # A new exhausted polling attempt for the same image is a new incident.
    errors = [{"type": "ImageBuildPollingError", "message": "Image build status unavailable [incident=" + "c" * 32 + "] build=example"}]
    assert watcher.failure_incident(errors, {"id": "x"}) != watcher.failure_incident([error], {"id": "x"})
