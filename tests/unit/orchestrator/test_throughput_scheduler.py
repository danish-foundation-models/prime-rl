import math
from collections import Counter

import pytest

from prime_rl.orchestrator.throughput_scheduler import ThroughputScheduler, allocate_group_budgets

SOURCES = [("fast", "fast", 0.4, 8), ("medium", "medium", 0.3, 8), ("slow", "slow", 0.3, 8)]


def test_consumption_tracks_targets_without_duplicates():
    scheduler = ThroughputScheduler(SOURCES, 256)
    rows = [(f"{name}-{i}", name) for name, *_ in SOURCES for i in range(500)]
    consumed = set()
    for _ in range(10):
        selected = scheduler.select(rows, 64)
        assert len(selected) == len(set(selected)) == 64
        assert not consumed.intersection(selected)
        names = dict(rows)
        scheduler.consume([names[i] for i in selected])
        consumed.update(selected)
        rows = [r for r in rows if r[0] not in set(selected)]
    assert sum(scheduler.served.values()) == 640
    for name, _, share, _ in SOURCES:
        assert abs(scheduler.served[name] / 640 - share) < 0.003


def test_stragglers_do_not_hold_a_small_admission_quota():
    scheduler = ThroughputScheduler(SOURCES, 256)
    assert scheduler.choose_sources({}, {"fast": 48, "medium": 40, "slow": 40}, 256)
    for _ in range(15):
        scheduler.observe_duration("slow", 3000)
        scheduler.observe_duration("fast", 30)
    chosen = scheduler.choose_sources({}, {"fast": 80, "medium": 80, "slow": 80}, 256)
    assert chosen == {"slow"}


def test_bounded_inventory_and_unavailable_families():
    scheduler = ThroughputScheduler(SOURCES, 256)
    assert not scheduler.choose_sources({"fast": 256}, {}, 256)
    assert scheduler.last_block == "ready_inventory"
    assert not scheduler.choose_sources({}, {}, 256, buffered_tokens=100, token_limit=100)
    assert scheduler.last_block == "buffered_tokens"
    assert not scheduler.choose_sources({}, {"slow": 4096}, 256)
    assert scheduler.last_block == "unfinished_inventory"
    assert scheduler.select([("x", "fast")], 1) == ["x"]
    assert scheduler.select([], 64) == []


def test_resume_preserves_mix_and_estimates():
    scheduler = ThroughputScheduler(SOURCES, 256)
    scheduler.consume(["fast"] * 40 + ["slow"] * 3)
    scheduler.observe_duration("slow", 2000)
    scheduler.observe_yield("medium", 8, 3)
    restored = ThroughputScheduler(SOURCES, 256)
    restored.load_state_dict(scheduler.state_dict())
    candidates = [(f"{name}{i}", name) for name, *_ in SOURCES for i in range(100)]
    assert restored.select(candidates, 64) == scheduler.select(candidates, 64)
    assert restored.choose_sources({}, {}, 256) == scheduler.choose_sources({}, {}, 256)
    assert restored.metrics() == scheduler.metrics()


def test_variants_do_not_multiply_family_target():
    scheduler = ThroughputScheduler([("a1", "a", 1, 8), ("a2", "a", 1, 8), ("b", "b", 2, 8)], 128)
    rows = [(f"{n}{i}", n) for n in ("a1", "a2", "b") for i in range(100)]
    selected = scheduler.select(rows, 64)
    counts = Counter(dict(rows)[i] for i in selected)
    assert counts == {"a1": 16, "a2": 16, "b": 32}


def test_bad_resume_sources_rejected():
    scheduler = ThroughputScheduler(SOURCES, 256)
    state = scheduler.state_dict()
    state["served"] = {"unknown": 1}
    with pytest.raises(ValueError):
        scheduler.load_state_dict(state)


CATEGORY_TARGETS = {"swe": 72, "terminal": 40, "sql": 8, "search": 8}
CATEGORY_SOURCES = [(n, n, 1, 8) for n in CATEGORY_TARGETS]


def category_scheduler():
    return ThroughputScheduler(CATEGORY_SOURCES, 512, category_targets=CATEGORY_TARGETS, batch_size=128)


def category_rows(**counts):
    return [(f"{n}-{i}", n) for n, count in counts.items() for i in range(count)]


def test_category_batch_target_and_unavailable_category():
    scheduler = category_scheduler()
    rows = category_rows(swe=300, terminal=300, sql=300, search=300)
    selected = scheduler.select(rows, 128)
    assert Counter(dict(rows)[i] for i in selected) == CATEGORY_TARGETS
    # Fast SQL cannot fill the batch when the terminal category is missing.
    assert scheduler.select(category_rows(swe=300, sql=300, search=300), 128) == []
    assert scheduler.batch_shortfall["terminal"] == 38
    # All categories may be represented by a single ready source; no taskset barrier.
    relaxed = category_rows(swe=74, terminal=38, sql=8, search=8)
    assert len(scheduler.select(relaxed, 128)) == 128


def test_category_surplus_cannot_accumulate_forever():
    scheduler = category_scheduler()
    for _ in range(4):
        rows = category_rows(swe=74, terminal=38, sql=8, search=8)
        ids = scheduler.select(rows, 128)
        assert len(ids) == 128
        scheduler.consume([dict(rows)[i] for i in ids])
    assert scheduler.family_served["swe"] - scheduler.total_served * scheduler.family_weights["swe"] == 8
    assert scheduler.select(rows, 128) == []
    # Enough terminal completions recover the target without resetting the debt.
    rows = category_rows(swe=72, terminal=42, sql=8, search=8)
    ids = scheduler.select(rows, 128)
    assert len(ids) == 128
    scheduler.consume([dict(rows)[i] for i in ids])
    for cat, weight in scheduler.family_weights.items():
        assert scheduler.family_served[cat] - scheduler.total_served * weight <= 8


def test_category_refill_survives_oversupplied_ready_buffer():
    scheduler = category_scheduler()
    # Already-running fast jobs can overshoot: still admit the missing categories.
    assert scheduler.choose_sources({"sql": 600}, {}, 256) - {"sql"}
    assert not scheduler.choose_sources({"sql": 600}, {}, 256, buffered_tokens=100, token_limit=100)
    state = scheduler.state_dict()
    other = ThroughputScheduler(
        CATEGORY_SOURCES, 512, category_targets={"swe": 64, "terminal": 48, "sql": 8, "search": 8}, batch_size=128
    )
    with pytest.raises(ValueError, match="balance configuration"):
        other.load_state_dict(state)


def test_mimo_clipped_oversampling_equation():
    demand = {"fast": 4.0, "medium": 2.0, "slow": 1.0}
    duration = {"fast": 10.0, "medium": 100.0, "slow": 1000.0}
    budgets = allocate_group_budgets(demand, duration, 3.0, 0.0, 8.0)
    ratios = {n: budgets[n] / demand[n] - 1 for n in demand}
    assert sum(demand[n] * ratios[n] for n in demand) / sum(demand.values()) == pytest.approx(3.0)
    assert all(0 <= p <= 8 for p in ratios.values())
    assert ratios["slow"] > ratios["fast"]
    assert ratios["slow"] == pytest.approx(8.0)


def test_mimo_target_deficit_blend_and_smooth_round_robin():
    scheduler = category_scheduler()
    scheduler.startup_dispatched = 10000
    _, empty = scheduler.scheduling_plan({}, 256)
    _, ready = scheduler.scheduling_plan({"terminal": 40}, 256)
    assert ready["terminal"] == pytest.approx(empty["terminal"] / 2)
    assert ready["swe"] == empty["swe"]
    launches = Counter(next(iter(scheduler.choose_sources({}, {}, 256))) for _ in range(160))
    assert launches == {"swe": 90, "terminal": 50, "sql": 10, "search": 10}
    state = scheduler.state_dict()
    restored = category_scheduler()
    restored.load_state_dict(state)
    assert [restored.choose_sources({"terminal": 40}, {}, 256) for _ in range(50)] == [
        scheduler.choose_sources({"terminal": 40}, {}, 256) for _ in range(50)
    ]


def test_mimo_budget_tracks_cap_without_ignoring_group_tails():
    scheduler = category_scheduler()
    small, _ = scheduler.scheduling_plan({}, 128)
    large, _ = scheduler.scheduling_plan({}, 512)
    assert sum(large.values()) > sum(small.values())
    assert sum(large.values()) * 8 == pytest.approx(1024)
    exhausted = {n: 8 * max(1, math.ceil(b)) for n, b in small.items()}
    assert not scheduler.choose_sources({}, exhausted, 128)
    assert scheduler.last_block == "source_group_budget"
    # A lower cap drains existing reservations; admitting elsewhere does not override permits.
    assert scheduler.choose_sources({}, {"swe": exhausted["swe"]}, 128) - {"swe"}


def test_coarse_budgets_pool_low_yield_members_without_starving_category():
    scheduler = ThroughputScheduler(
        [("working", "terminal", 1, 8), ("hard", "terminal", 1, 8), ("swe", "swe", 2, 8)],
        512,
        category_targets={"terminal": 1, "swe": 1},
        batch_size=128,
    )
    scheduler.yield_rate["working"] = 1
    scheduler.yield_rate["hard"] = 0
    budgets, weights = scheduler.scheduling_plan({}, 256)
    assert set(budgets) == {"terminal", "swe"}
    assert weights["terminal"] == pytest.approx(2 * weights["swe"])
    scheduler.startup_dispatched = 10000
    selected = Counter(next(iter(scheduler.choose_sources({}, {}, 256))) for _ in range(120))
    assert selected == {"working": 40, "hard": 40, "swe": 40}
