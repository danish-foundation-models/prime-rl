import asyncio
import time
import uuid
from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.orchestrator.dispatcher import DispatcherMetrics, RolloutDispatcher
from prime_rl.orchestrator.types import GroupState, InflightRollout, Rollout


def make_dispatcher(*, replacements: int = 2, hedge_after: float | None = 10) -> RolloutDispatcher:
    dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
    dispatcher.max_group_replacements = replacements
    dispatcher.group_hedge_after_seconds = hedge_after
    dispatcher.groups = {}
    dispatcher.inflight = {}
    dispatcher.inflight_permits = 0
    dispatcher.metrics = DispatcherMetrics()
    return dispatcher


def make_group() -> GroupState:
    return GroupState(
        kind="train",
        env_name="slow",
        task_idx=7,
        rollouts_to_schedule=0,
        target_rollouts=8,
        emitted=7,
    )


def test_group_replacements_are_bounded():
    dispatcher = make_dispatcher(replacements=2)
    group = make_group()

    assert dispatcher._request_group_replacement(group)
    assert dispatcher._request_group_replacement(group)
    assert not dispatcher._request_group_replacement(group)
    assert group.replacements_started == 2
    assert group.replacements_pending == 2
    assert group.rollouts_to_schedule == 2


def test_old_final_member_queues_one_hedge():
    dispatcher = make_dispatcher()
    group_id = uuid.uuid4()
    group = make_group()
    dispatcher.groups[group_id] = group
    dispatcher.inflight[object()] = InflightRollout(
        kind="train",
        env_name="slow",
        group_id=group_id,
        policy_version=0,
        rollout_count=1,
        started_at=time.monotonic() - 11,
    )

    dispatcher._queue_tail_hedge()

    assert group.replacements_started == 1
    assert group.replacements_pending == 1
    assert group.rollouts_to_schedule == 1


def test_completing_hedged_group_cancels_late_sibling():
    async def run() -> None:
        dispatcher = make_dispatcher()
        dispatcher.out_q = asyncio.Queue()
        group_id = uuid.uuid4()
        group = make_group()
        dispatcher.groups[group_id] = group
        sibling = asyncio.create_task(asyncio.sleep(60))
        dispatcher.inflight[sibling] = InflightRollout(
            kind="train",
            env_name="slow",
            group_id=group_id,
            policy_version=0,
            rollout_count=1,
            is_replacement=True,
        )
        dispatcher.inflight_permits = 1
        meta = InflightRollout(
            kind="train",
            env_name="slow",
            group_id=group_id,
            policy_version=0,
            rollout_count=1,
        )
        rollout = Rollout(task=vf.TraceTask(type="Task", data=vf.TaskData(idx=7, prompt="test")))

        await dispatcher.emit_rollout(meta, group, rollout)

        assert group_id not in dispatcher.groups
        assert not dispatcher.inflight
        assert dispatcher.inflight_permits == 0
        assert sibling.cancelled()
        assert await dispatcher.out_q.get() is rollout

    asyncio.run(run())


def test_failed_member_stays_out_of_sink_and_queues_replacement():
    async def run() -> None:
        dispatcher = make_dispatcher()
        dispatcher.out_q = asyncio.Queue()
        dispatcher.train_source = SimpleNamespace(mixture=None)
        dispatcher.train_envs = SimpleNamespace(get=lambda _name: SimpleNamespace(requires_group_scoring=False))
        group_id = uuid.uuid4()
        group = make_group()
        dispatcher.groups[group_id] = group
        failed = Rollout(
            task=vf.TraceTask(type="Task", data=vf.TaskData(idx=7, prompt="test")),
            errors=[vf.Error(type="SandboxError", message="transient")],
            stop_condition="error",
        )
        task = asyncio.create_task(asyncio.sleep(0, result=failed))
        await task
        dispatcher.inflight[task] = InflightRollout(
            kind="train",
            env_name="slow",
            group_id=group_id,
            policy_version=0,
            rollout_count=1,
        )
        dispatcher.inflight_permits = 1

        await dispatcher.handle_completed_rollout(task)

        assert dispatcher.out_q.empty()
        assert group.emitted == 7
        assert group.rollouts_to_schedule == 1
        assert group.replacements_pending == 1
        assert group.replacements_started == 1

    asyncio.run(run())
