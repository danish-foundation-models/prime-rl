from collections import Counter

from prime_rl.orchestrator.dispatcher import DispatcherMetrics
from prime_rl.orchestrator.mixture import MixturePlanner


def test_batch_plans_preserve_groups_and_track_ratios_across_steps():
    planner = MixturePlanner(
        ratios={"fast": 0.75, "slow": 0.25},
        group_sizes={"fast": 4, "slow": 4},
        batch_size=8,
        max_inflight=32,
    )

    plans = [planner.plan_batch() for _ in range(4)]
    counts = Counter(name for plan in plans for name in plan)

    assert all(len(plan) == 2 for plan in plans)
    assert counts == {"fast": 6, "slow": 2}


def test_inventory_feedback_stops_overfilled_environment():
    planner = MixturePlanner(
        ratios={"fast": 0.75, "slow": 0.25},
        group_sizes={"fast": 4, "slow": 4},
        batch_size=8,
        max_inflight=32,
    )

    eligible = planner.environments_needing_work(
        pending={"fast": 20, "slow": 0},
        inflight={"fast": 4, "slow": 4},
        costs={"fast": 4, "slow": 4},
        available_permits=4,
    )

    assert eligible == ["slow"]


def test_dispatcher_lifecycle_metrics_are_dense_and_drained():
    metrics = DispatcherMetrics()
    metrics.record_launch(kind="train", env_name="slow", n=4)
    metrics.record_completion(kind="train", env_name="slow", n=3)
    metrics.record_error(kind="train", env_name="slow")

    first = metrics.drained(train_envs={"fast", "slow"}, eval_envs=set())
    second = metrics.drained(train_envs={"fast", "slow"}, eval_envs=set())

    assert first["dispatcher/launched/train"] == 4
    assert first["dispatcher/completed/slow"] == 3
    assert first["dispatcher/errored/slow"] == 1
    assert second["dispatcher/launched/train"] == 0
    assert second["dispatcher/completed/slow"] == 0
