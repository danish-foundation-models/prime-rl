from collections import Counter

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
