import asyncio
import time

import prime_rl.utils.logger as logger_module
from prime_rl.utils.async_utils import EventLoopStallWatchdog, safe_cancel


class _Recorder:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def test_watchdog_reports_a_blocked_loop_with_its_stack(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(logger_module, "get_logger", lambda: recorder)

    def blocking_call():
        time.sleep(0.6)

    async def main():
        task = asyncio.create_task(EventLoopStallWatchdog(threshold=0.2, interval=0.05).run())
        await asyncio.sleep(0.1)
        blocking_call()
        await asyncio.sleep(0.2)
        await safe_cancel(task)

    asyncio.run(main())
    stalled = [w for w in recorder.warnings if "stalled" in w]
    assert len(stalled) == 1
    assert "--- event loop" in stalled[0] and "blocking_call" in stalled[0]
    assert any("recovered" in w for w in recorder.warnings)
