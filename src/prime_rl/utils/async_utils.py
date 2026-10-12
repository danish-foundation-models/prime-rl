from __future__ import annotations

import asyncio
import sys
import threading
import traceback
from collections import deque
from time import perf_counter

import numpy as np
from pydantic import BaseModel


async def safe_cancel(task: asyncio.Task) -> None:
    """Safely cancels and awaits an asyncio.Task."""
    task.cancel()
    try:
        await task
    except BaseException:
        pass


async def safe_cancel_all(tasks: list[asyncio.Task]) -> None:
    """Safely cancels and awaits all asyncio.Tasks."""
    await asyncio.gather(*[safe_cancel(task) for task in tasks])


class EventLoopLagMonitor:
    """Monitors how busy the main event loop is by timing short sleeps."""

    def __init__(self, measure_interval: float = 0.1, max_measurements: int = 1000):
        assert measure_interval > 0 and max_measurements > 0
        self.measure_interval = measure_interval
        self.max_measurements = max_measurements
        self.lags: deque[float] = deque(maxlen=max_measurements)

    async def measure_lag(self) -> float:
        next_time = perf_counter() + self.measure_interval
        await asyncio.sleep(self.measure_interval)
        return perf_counter() - next_time

    async def run(self) -> None:
        """Loop measuring event-loop lag; run as a background task."""
        while True:
            self.lags.append(await self.measure_lag())


class EventLoopLagStats(BaseModel):
    """Snapshot of event-loop lag statistics."""

    min: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    max: float = 0.0
    n: int = 0

    @classmethod
    def from_monitor(cls, monitor: EventLoopLagMonitor) -> EventLoopLagStats:
        n = len(monitor.lags)
        if n == 0:
            return cls(n=0)
        arr = np.array(monitor.lags)
        return cls(
            min=float(arr.min()),
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            p90=float(np.percentile(arr, 90)),
            p99=float(np.percentile(arr, 99)),
            max=float(arr.max()),
            n=n,
        )


class EventLoopStallWatchdog:
    """Logs the stacks of every thread when the event loop stops ticking.

    The lag monitor only reports after the loop recovers, and only to W&B. A
    watchdog thread reads a heartbeat the loop refreshes; past ``threshold``
    seconds it logs where each thread is (the loop thread's frame names the
    blocking call, another thread's frame names a GIL holder) and, on recovery,
    how long the stall lasted.
    """

    def __init__(self, threshold: float = 10.0, interval: float = 0.5):
        self.threshold = threshold
        self.interval = interval
        self.beat = perf_counter()
        self.loop_thread_id: int | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    async def run(self) -> None:
        """Heartbeat task; starts the watchdog thread on first run."""
        self.loop_thread_id = threading.get_ident()
        self.beat = perf_counter()
        if self.thread is None:
            self.thread = threading.Thread(target=self._watch, name="event-loop-stall-watchdog", daemon=True)
            self.thread.start()
        try:
            while True:
                self.beat = perf_counter()
                await asyncio.sleep(self.interval)
        finally:
            self.stop_event.set()

    def _watch(self) -> None:
        from prime_rl.utils.logger import get_logger

        stalled_since: float | None = None
        while not self.stop_event.wait(self.interval):
            silent = perf_counter() - self.beat
            if silent > self.threshold and stalled_since is None:
                stalled_since = self.beat
                get_logger().warning(f"Event loop stalled for {silent:.1f}s; thread stacks:\n{self._stacks()}")
            elif silent <= self.threshold and stalled_since is not None:
                get_logger().warning(f"Event loop recovered after a {self.beat - stalled_since:.1f}s stall")
                stalled_since = None

    def _stacks(self) -> str:
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        parts = []
        for ident, frame in sys._current_frames().items():
            if ident == threading.get_ident():
                continue
            label = "event loop" if ident == self.loop_thread_id else names.get(ident, str(ident))
            stack = "".join(traceback.format_stack(frame)[-8:])
            parts.append(f"--- {label}\n{stack}")
        return "".join(parts)
