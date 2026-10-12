import time

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator.concurrency import KV_CEILING_FRACTION, KV_CEILING_RECOVERY, ConcurrencyController, EngineLoadSample


def sample(kv_usage: float, *, preemptions: int = 0) -> EngineLoadSample:
    return EngineLoadSample(
        engine_id="e0", role=None, kv_capacity_tokens=1_000_000, max_model_len=131072,
        kv_usage=kv_usage, running=10, waiting=0, waiting_capacity=0, preemptions_delta=preemptions,
    )


def controller(start: int = 800):
    inflight = {"n": start}
    cancelled = []
    c = ConcurrencyController(ConcurrencyConfig(initial_inflight=start, min_inflight=64, max_inflight=1024), fallback_cost=131072)
    c.bind(set_limit=lambda n: None, get_inflight=lambda: inflight["n"], on_overload=cancelled.append)
    return c, inflight, cancelled


def grow(c, inflight, completions: int) -> None:
    """Completions while the engines are clear and the cap binds."""
    for _ in range(completions):
        c.observe([sample(0.3)])
        inflight["n"] = c.max_inflight - 1
        c.record_episode("env", "train", tokens=1000, duration=60.0)


def test_kv_trim_above_the_hard_cap_drains_instead_of_cancelling():
    c, inflight, cancelled = controller(800)
    c.observe([sample(0.95)])
    assert c.max_inflight < 800
    assert cancelled == []


def test_growth_after_a_kv_trim_stops_below_the_pool_that_overflowed():
    c, inflight, _ = controller(800)
    c.observe([sample(0.95)])
    ceiling = KV_CEILING_FRACTION * 800
    start = c.turnover
    for _ in range(30):
        grow(c, inflight, 100)
        assert c.max_inflight <= int(c.kv_ceiling) + 1
        assert c.kv_ceiling <= ceiling * KV_CEILING_RECOVERY ** (c.turnover - start) + 1e-6
    assert c.max_inflight >= int(ceiling * 0.95)  # it grows back up to the ceiling
    assert c.max_inflight < 800  # but stays below the pool that overflowed for several turnovers


def test_the_ceiling_rises_slowly_while_the_engines_stay_clear():
    c, inflight, _ = controller(800)
    c.observe([sample(0.95)])
    first = c.kv_ceiling
    grow(c, inflight, 4000)
    assert first < c.kv_ceiling <= first * KV_CEILING_RECOVERY ** c.turnover


def test_growth_is_gentler_than_a_quarter_per_turnover():
    c, inflight, _ = controller(200)
    grow(c, inflight, 200)  # about one turnover
    assert c.max_inflight <= 200 * 1.12


def test_preemptions_still_cancel_the_excess():
    c, inflight, cancelled = controller(800)
    c.observe([sample(0.5, preemptions=3)])
    assert cancelled and cancelled[0] > 0
