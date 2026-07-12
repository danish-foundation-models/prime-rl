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


def test_service_time_feedback_redirects_free_capacity_to_slow_environment():
    planner = MixturePlanner(
        ratios={"fast": 0.75, "slow": 0.25},
        group_sizes={"fast": 4, "slow": 4},
        batch_size=8,
        max_inflight=32,
        service_time_alpha=0.2,
        max_supply_multiplier=2,
    )

    planner.observe_completion("fast", service_seconds=10, tokens_per_rollout=100)
    planner.observe_completion("slow", service_seconds=40, tokens_per_rollout=400)

    eligible = planner.environments_needing_work(
        pending={"fast": 0, "slow": 0},
        inflight={"fast": 16, "slow": 8},
        costs={"fast": 4, "slow": 4},
        available_permits=4,
    )

    assert planner.choose_environment(eligible=eligible, inflight={"fast": 16, "slow": 8}) == "slow"

    eligible = planner.environments_needing_work(
        pending={"fast": 24, "slow": 0},
        inflight={"fast": 0, "slow": 16},
        costs={"fast": 4, "slow": 4},
        available_permits=4,
    )

    assert eligible == []


def test_unobserved_environment_inherits_slowest_observed_service_time():
    planner = MixturePlanner(
        ratios={"fast": 0.5, "unseen": 0.5},
        group_sizes={"fast": 4, "unseen": 4},
        batch_size=8,
        max_inflight=32,
        service_time_alpha=0.2,
    )

    planner.observe_completion("fast", service_seconds=20, tokens_per_rollout=200)

    assert planner.service_seconds == {"fast": 20, "unseen": 20}
    assert planner.tokens_per_rollout == {"fast": 200, "unseen": 200}


def test_failures_reduce_only_generate_ahead_headroom():
    planner = MixturePlanner(
        ratios={"healthy": 0.75, "failing": 0.25},
        group_sizes={"healthy": 4, "failing": 4},
        batch_size=8,
        max_inflight=32,
        service_time_alpha=0.2,
        success_rate_alpha=0.5,
        max_supply_multiplier=2,
    )

    for _ in range(4):
        planner.observe_completion(
            "failing",
            service_seconds=1,
            tokens_per_rollout=0,
            success=False,
        )

    assert planner.inventory_limits["failing"] == 8
    assert planner.supply_limits["failing"] == 16
    assert planner.effective_supply_limits["failing"] == 8
    assert planner.effective_supply_limits["healthy"] == 48


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
