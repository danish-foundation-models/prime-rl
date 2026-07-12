"""RolloutDispatcher: schedules rollouts under a shared permit counter.

- Capacity (``max_inflight_rollouts``) is shared across train + eval.
  A group-scoring task that runs N rollouts in one call reserves N permits.
- Optional rate limiting via ``AsyncLimiter(tasks_per_minute, 60)``.
- Every ordinary dispatched rollout reaches ``out_q`` once. When bounded
  complete-group replacement is configured, failed attempts stay out of the
  sink and late hedged siblings are cancelled after the group fills.
- ``DispatcherMode.PREFER_TRAIN`` / ``PREFER_EVAL`` controls which kind to
  schedule next. Transitions are level-triggered (driven by the eval
  source's emptiness), so in-flight rollouts of the opposite kind drain
  naturally on either side of an eval boundary.
- ``on_version_pending`` (called by the watcher before the engines pause for
  the weight update) bumps ``off_policy_steps`` on in-flight train rollouts and
  drops groups past ``max_off_policy_steps``.
  Eval rollouts are measurements for the policy version they started with,
  so they are allowed to finish even if training advances. Train rollouts
  sampled from a frozen model never age — their sampler doesn't change
  with policy updates.
  Cancellations surface as synthetic ``Cancelled`` markers so the sink's
  count-to-``group_size`` finalization still fires.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import verifiers.v1 as vf
from aiolimiter import AsyncLimiter

from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    GroupState,
    InflightRollout,
    Policy,
    Rollout,
    RolloutKind,
)
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool, client_identity
from prime_rl.utils.logger import get_logger


class DispatcherMode(Enum):
    """Which kind of work the dispatcher schedules next."""

    PREFER_TRAIN = auto()
    PREFER_EVAL = auto()


@dataclass
class DispatcherMetrics:
    """Per-tick drain counters for the orchestrator's periodic log.
    ``drained()`` returns the current values and clears them; point-in-time
    gauges live on ``RolloutDispatcher.gauges`` instead."""

    cancelled_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    errored_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    launched_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    completed_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    replacements_by_env: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_launch(self, *, kind: Literal["train", "eval"], env_name: str, n: int) -> None:
        self.launched_by_kind_env[(kind, env_name)] += n

    def record_completion(self, *, kind: Literal["train", "eval"], env_name: str, n: int) -> None:
        self.completed_by_kind_env[(kind, env_name)] += n

    def record_cancellation(self, *, kind: Literal["train", "eval"], env_name: str, n: int = 1) -> None:
        self.cancelled_by_kind_env[(kind, env_name)] += n

    def record_error(self, *, kind: Literal["train", "eval"], env_name: str) -> None:
        self.errored_by_kind_env[(kind, env_name)] += 1

    def record_replacement(self, *, env_name: str) -> None:
        self.replacements_by_env[env_name] += 1

    def drained(self, *, train_envs: set[str], eval_envs: set[str]) -> dict[str, float]:
        """Return per-tick counters and clear them. Emits the full pre-
        registered key set every tick (zero when no activity) so the wandb
        time axis stays dense and ``define_metric`` lines up."""
        out: dict[str, float] = {}
        for kind in ("train", "eval"):
            envs = train_envs if kind == "train" else eval_envs
            launched_total = sum(self.launched_by_kind_env.get((kind, e), 0) for e in envs)
            completed_total = sum(self.completed_by_kind_env.get((kind, e), 0) for e in envs)
            cancelled_total = sum(self.cancelled_by_kind_env.get((kind, e), 0) for e in envs)
            errored_total = sum(self.errored_by_kind_env.get((kind, e), 0) for e in envs)
            out[f"dispatcher/launched/{kind}"] = float(launched_total)
            out[f"dispatcher/completed/{kind}"] = float(completed_total)
            out[f"dispatcher/cancelled/{kind}"] = float(cancelled_total)
            out[f"dispatcher/errored/{kind}"] = float(errored_total)
        for env in train_envs | eval_envs:
            out[f"dispatcher/launched/{env}"] = float(
                self.launched_by_kind_env.get(("train", env), 0) + self.launched_by_kind_env.get(("eval", env), 0)
            )
            out[f"dispatcher/completed/{env}"] = float(
                self.completed_by_kind_env.get(("train", env), 0) + self.completed_by_kind_env.get(("eval", env), 0)
            )
            out[f"dispatcher/cancelled/{env}"] = float(
                self.cancelled_by_kind_env.get(("train", env), 0) + self.cancelled_by_kind_env.get(("eval", env), 0)
            )
            out[f"dispatcher/errored/{env}"] = float(
                self.errored_by_kind_env.get(("train", env), 0) + self.errored_by_kind_env.get(("eval", env), 0)
            )
            out[f"dispatcher/replacements/{env}"] = float(self.replacements_by_env.get(env, 0))
        self.launched_by_kind_env.clear()
        self.completed_by_kind_env.clear()
        self.cancelled_by_kind_env.clear()
        self.errored_by_kind_env.clear()
        self.replacements_by_env.clear()
        return out

    @staticmethod
    def drain_keys(*, train_envs: set[str], eval_envs: set[str]) -> list[str]:
        """Full set of keys ``drained`` may emit; used by the periodic
        logger for ``wandb.define_metric``."""
        keys = [
            "dispatcher/launched/train",
            "dispatcher/launched/eval",
            "dispatcher/completed/train",
            "dispatcher/completed/eval",
            "dispatcher/cancelled/train",
            "dispatcher/cancelled/eval",
            "dispatcher/errored/train",
            "dispatcher/errored/eval",
        ]
        for env in train_envs | eval_envs:
            keys.append(f"dispatcher/launched/{env}")
            keys.append(f"dispatcher/completed/{env}")
            keys.append(f"dispatcher/cancelled/{env}")
            keys.append(f"dispatcher/errored/{env}")
            keys.append(f"dispatcher/replacements/{env}")
        return keys


class RolloutDispatcher:
    """``await dispatcher.start()`` runs the dispatch loop until ``stop()``.
    Pulls examples from ``TrainSource`` / ``EvalSource``, schedules
    rollouts under shared capacity, and emits ``Rollout``\\ s to
    ``out_q``. The watcher drives ``on_version_pending`` for off-policy
    cancellation; the orchestrator triggers eval epochs."""

    def __init__(
        self,
        *,
        train_envs: TrainEnvs,
        eval_envs: EvalEnvs | None,
        train_source: TrainSource,
        eval_source: EvalSource | None,
        policy_pool: InferencePool,
        policy: Policy,
        max_inflight_rollouts: int,
        tasks_per_minute: float | None,
        max_off_policy_steps: int,
        max_group_replacements: int = 0,
        group_hedge_after_seconds: float | None = None,
    ) -> None:
        self.policy = policy
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        # Train rollouts go to the env sampler's pool; eval always
        # evaluates the policy.
        self.policy_pool = policy_pool
        self.train_source = train_source
        self.eval_source = eval_source
        self.max_off_policy_steps = max_off_policy_steps
        self.max_group_replacements = max_group_replacements
        self.group_hedge_after_seconds = group_hedge_after_seconds

        self.max_inflight = max_inflight_rollouts
        self.inflight_permits = 0
        self.rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(tasks_per_minute, time_period=60) if tasks_per_minute else None
        )

        self.inflight: dict[asyncio.Task, InflightRollout] = {}
        self.groups: dict[uuid.UUID, GroupState] = {}

        # Bounded so the dispatcher backpressures on a slow sink
        self.out_q: asyncio.Queue[Rollout] = asyncio.Queue(maxsize=max(8, self.max_inflight))

        self.mode: DispatcherMode = DispatcherMode.PREFER_TRAIN
        # Set by the orchestrator after the final train step; pipeline then
        # winds down without scheduling new train rollouts
        self.train_scheduling_disabled: bool = False
        self.metrics = DispatcherMetrics()

        # Orchestrator-owned gate. When clear, ``fill_inflight`` returns
        # without scheduling new groups. The dispatcher itself doesn't know
        # *why* — the orchestrator toggles this based on step / policy lead.
        self.dispatch_allowed = asyncio.Event()
        self.dispatch_allowed.set()

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

    def _train_pool_for(self, env_name: str) -> tuple[InferencePool, str, bool]:
        """``(pool, model_name, is_live)`` for *train* rollouts of this env —
        the env sampler's pool. (Eval always uses the policy.)"""
        sampler = self.train_envs.get(env_name).sampler
        if sampler.samples_from_live_policy:
            return sampler.pool, self.policy.model_name, True
        return sampler.pool, sampler.pool.model_name, False

    @property
    def inflight_train_count(self) -> int:
        return sum(m.rollout_count for m in self.inflight.values() if m.kind == "train")

    @property
    def inflight_eval_count(self) -> int:
        return sum(m.rollout_count for m in self.inflight.values() if m.kind == "eval")

    @property
    def available_permits(self) -> int:
        return self.max_inflight - self.inflight_permits

    @property
    def inflight_by_env(self) -> dict[tuple[RolloutKind, str], int]:
        counts: dict[tuple[RolloutKind, str], int] = defaultdict(int)
        for meta in self.inflight.values():
            counts[(meta.kind, meta.env_name)] += meta.rollout_count
        return dict(counts)

    @property
    def queued_eval_examples(self) -> int:
        return len(self.eval_source) if self.eval_source is not None else 0

    @property
    def eval_has_work(self) -> bool:
        """Eval has work while its source queue is non-empty OR any opened eval group still has
        rollouts to schedule. An example leaves ``eval_source`` when its group opens
        (``next_fresh_group``), but its ``group_size`` rollouts dispatch one at a time across
        ``fill_inflight`` passes — so the queue can be empty while a group is still mid-schedule."""
        return bool(self.eval_source) or any(
            g.kind == "eval" and g.rollouts_to_schedule > 0 for g in self.groups.values()
        )

    @property
    def is_idle(self) -> bool:
        """True once nothing is in flight, no eval work remains (queued *or* a partly-scheduled eval
        group), and ``out_q`` is empty — the pipeline has fully drained."""
        return not self.inflight and not self.eval_has_work and self.out_q.empty()

    def disable_train_scheduling(self) -> None:
        """Stop scheduling new train rollouts; in-flight train + any
        triggered eval drain naturally."""
        self.train_scheduling_disabled = True

    @property
    def max_off_policy_level(self) -> int:
        steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return max(steps) if steps else 0

    @property
    def mean_off_policy_level(self) -> float:
        steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return sum(steps) / len(steps) if steps else 0.0

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Single dispatch loop: schedule, wait, collect, repeat."""
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                await self.fill_inflight()
                if not self.inflight:
                    # No work — sleep briefly. Eval triggers from the
                    # orchestrator wake the next iteration via a mode flip
                    try:
                        await asyncio.wait_for(self.stopped.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    continue

                done, _pending = await asyncio.wait(
                    list(self.inflight.keys()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.5,  # wake periodically to re-check fill (mode flips)
                )
                for task in done:
                    await self.handle_completed_rollout(task)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        await self.cancel_inflight_rollouts()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def on_version_pending(self, step: int) -> None:
        """Bump off-policy counters and drop groups past
        ``max_off_policy_steps`` (drop_group emits ``Cancelled`` markers so
        the sink still finalizes the partial group). Eval rollouts are not
        aged because they are tied to their start-time policy version.

        Runs *before* the inference engines are paused for the weight update so
        the resulting aborts are processed while the engine is still stepping —
        otherwise the orphaned KV transfers crash the decode engine on resume
        (see ``WeightWatcher.apply_policy_update``)."""
        stale_groups: set[uuid.UUID] = set()
        cancelled = 0
        for meta in self.inflight.values():
            if meta.kind != "train":
                continue
            # Frozen-sourced rollouts never go stale — their sampler doesn't
            # change with policy updates.
            if not self.train_envs.get(meta.env_name).sampler.samples_from_live_policy:
                continue
            meta.off_policy_steps += 1
            if meta.off_policy_steps > self.max_off_policy_steps:
                stale_groups.add(meta.group_id)

        for gid in stale_groups:
            removed = await self.drop_group(gid)
            cancelled += removed

        if cancelled:
            get_logger().warning(
                f"Cancelled {cancelled} train rollouts past max_off_policy_steps={self.max_off_policy_steps}. "
                "Consider increasing it to avoid this."
            )

    async def on_new_version(self, step: int) -> None:
        """No-op: the dispatcher drains in ``on_version_pending`` (pre-pause)."""

    async def fill_inflight(self) -> None:
        """Schedule new rollouts up to ``max_inflight``, honoring
        ``self.mode``. Eval scheduling ignores the orchestrator's dispatch
        gate (evals are version-pinned measurements); only train scheduling
        respects it. When ``PREFER_EVAL``'s source exhausts we flip back to
        ``PREFER_TRAIN`` so the eval tail drains alongside fresh train."""
        while True:
            if self.available_permits <= 0:
                return

            if self.mode == DispatcherMode.PREFER_EVAL:
                # PREFER_EVAL is only entered when the orchestrator triggers
                # eval, which requires ``eval_source`` to be configured
                assert self.eval_source is not None
                if not self.eval_has_work:
                    # Eval source + all eval groups fully dispatched. Flip
                    # to PREFER_TRAIN so any remaining permits go to train
                    # while the in-flight eval tail completes naturally
                    self.switch_mode(DispatcherMode.PREFER_TRAIN, reason="the eval queue drained")
                    continue
                scheduled = await self.try_schedule("eval")
                if not scheduled:
                    return
            else:  # PREFER_TRAIN — respects the orchestrator's dispatch gate
                if not self.dispatch_allowed.is_set():
                    return
                scheduled = await self.try_schedule("train")
                if not scheduled:
                    return

    def switch_mode(self, new_mode: DispatcherMode, *, reason: str) -> None:
        if new_mode == self.mode:
            return
        prefer = "eval" if new_mode == DispatcherMode.PREFER_EVAL else "train"
        get_logger().info(f"Switching dispatcher mode to prefer {prefer} rollouts because {reason}")
        self.mode = new_mode

    async def try_schedule(self, kind: RolloutKind) -> bool:
        """Schedule one rollout of ``kind``: prefer continuing an existing
        group (keeps prefix-cache hits); otherwise open a fresh group from
        the corresponding source. Returns False if nothing could be
        scheduled."""
        if kind == "train" and self.train_scheduling_disabled:
            return False
        envs = self.train_envs if kind == "train" else self.eval_envs
        if envs is None:
            return False

        if kind == "train":
            self._queue_tail_hedge()

        for gid, group in list(self.groups.items()):
            if group.kind != kind or group.rollouts_to_schedule <= 0:
                continue
            env = envs.get(group.env_name)
            cost = group.rollouts_to_schedule if env.requires_group_scoring else 1
            if cost <= self.available_permits:
                return await self.schedule_group_rollout(gid, group)

        fresh = self.next_fresh_group(kind, envs)
        if fresh is None:
            return False
        gid = uuid.uuid4()
        self.groups[gid] = fresh
        return await self.schedule_group_rollout(gid, fresh)

    def _request_group_replacement(self, group: GroupState) -> bool:
        if group.replacements_started >= self.max_group_replacements:
            return False
        group.replacements_started += 1
        group.replacements_pending += 1
        group.rollouts_to_schedule += 1
        return True

    def _queue_tail_hedge(self) -> None:
        if self.group_hedge_after_seconds is None or self.max_group_replacements == 0:
            return
        now = time.monotonic()
        for group_id, group in self.groups.items():
            if (
                group.kind != "train"
                or group.emitted != group.target_rollouts - 1
                or group.rollouts_to_schedule > 0
            ):
                continue
            outstanding = [meta for meta in self.inflight.values() if meta.group_id == group_id]
            if not outstanding or any(meta.is_replacement for meta in outstanding):
                continue
            oldest = min(meta.started_at for meta in outstanding)
            if now - oldest < self.group_hedge_after_seconds:
                continue
            if self._request_group_replacement(group):
                get_logger().info(
                    "Hedging rollout tail | group=%s env=%s completed=%d/%d age=%.1fs",
                    str(group_id)[:8],
                    group.env_name,
                    group.emitted,
                    group.target_rollouts,
                    now - oldest,
                )
            return

    def next_fresh_group(self, kind: RolloutKind, envs) -> GroupState | None:
        """Pop the next example from the corresponding source and wrap it in
        a ``GroupState``. Returns ``None`` if the source is empty or the
        picked env's permit cost doesn't fit."""
        if kind == "train":
            source = self.train_source
        else:
            assert self.eval_source is not None
            source = self.eval_source
        if kind == "train":
            counts = self.inflight_by_env
            example = source.next_example(
                self.available_permits,
                {env.name: counts.get(("train", env.name), 0) for env in self.train_envs},
            )
        else:
            example = source.next_example(self.available_permits)
        if example is None:
            return None

        env_name = example["env_name"]
        group_size = envs.get(env_name).config.group_size
        eval_step: int | None = example.get("eval_step") if kind == "eval" else None

        return GroupState(
            kind=kind,
            env_name=env_name,
            task_idx=example["task_idx"],
            rollouts_to_schedule=group_size,
            target_rollouts=group_size,
            eval_step=eval_step,
            policy_version_at_start=self.policy.version,
        )

    async def schedule_group_rollout(self, group_id: uuid.UUID, group: GroupState) -> bool:
        """Dispatch one ``run_rollout`` / ``run_group`` task for this group.

        Returns False only if we couldn't even schedule one rollout (no clients
        ready, no permits). Returns True after issuing one task — the caller
        loops to keep scheduling.
        """
        # Train rollouts use the env sampler's pool via the
        # renderer/token train client. Eval always evaluates the policy and
        # goes through the eval client (chat-completions) — the same path the
        # legacy orchestrator used, so eval scores stay comparable.
        if group.kind == "eval":
            pool, model_name = self.policy_pool, self.policy.model_name
            live_sourced = True
        else:
            pool, model_name, live_sourced = self._train_pool_for(group.env_name)

        # Pin a single client per group to keep prefix-cache hits
        if group.pinned_client is None:
            if group.kind == "eval":
                client = await pool.get_eval_client()
            else:
                load = Counter(
                    client_identity(m.client_config) for m in self.inflight.values() if m.client_config is not None
                )
                client = await pool.select_train_client(load)
            if group_id not in self.groups:
                return False
            group.pinned_client = client
        else:
            client = group.pinned_client

        env_collection = self.train_envs if group.kind == "train" else self.eval_envs
        if env_collection is None:
            return False
        env = env_collection.get(group.env_name)
        is_replacement = group.replacements_pending > 0
        # Frozen-sourced train rollouts hit a frozen pool; salting per policy
        # version would invalidate its prefix cache every weight update for
        # no reason.
        if live_sourced:
            cache_salt = str(group.policy_version_at_start)
        else:
            cache_salt = None

        if env.requires_group_scoring:
            permits = group.rollouts_to_schedule
            group.rollouts_to_schedule = 0
            await self.acquire(permits)
            task: asyncio.Task = asyncio.create_task(
                env.run_group(
                    client=client,
                    task_idx=group.task_idx,
                    model_name=model_name,
                    group_size=permits,
                    cache_salt=cache_salt,
                )
            )
        else:
            permits = 1
            group.rollouts_to_schedule -= 1
            if is_replacement:
                group.replacements_pending -= 1
            await self.acquire(permits)
            task = asyncio.create_task(
                env.run_rollout(
                    client=client,
                    task_idx=group.task_idx,
                    model_name=model_name,
                    cache_salt=cache_salt,
                )
            )

        self.inflight[task] = InflightRollout(
            kind=group.kind,
            env_name=group.env_name,
            group_id=group_id,
            policy_version=group.policy_version_at_start,
            rollout_count=permits,
            started_at=time.monotonic(),
            is_replacement=is_replacement,
            client_config=client,
            eval_step=group.eval_step,
        )
        self.metrics.record_launch(kind=group.kind, env_name=group.env_name, n=permits)
        if is_replacement:
            self.metrics.record_replacement(env_name=group.env_name)
        return True

    async def acquire(self, n: int) -> None:
        """Reserve ``n`` permits + rate-limit each one. Caller must precheck
        ``available_permits >= n``; this is not a blocking acquire."""
        for _ in range(n):
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire()
            self.inflight_permits += 1

    def release(self, n: int) -> None:
        self.inflight_permits -= n

    async def handle_completed_rollout(self, task: asyncio.Task) -> None:
        """Emit completed rollouts, or replace failed complete-group members.
        Task exceptions synthesize error rollouts before the same policy is applied.
        Cancelled tasks (popped by ``drop_group``) raise ``CancelledError``
        and are discarded — ``drop_group`` already emitted their markers.
        """
        meta = self.inflight.pop(task, None)
        if meta is None:
            return  # already handled by drop_group / cancel_inflight_rollouts
        self.release(meta.rollout_count)
        group = self.groups.get(meta.group_id)

        is_synth_exception = False
        try:
            result = task.result()
            rollouts: list[Rollout] = result if isinstance(result, list) else [result]
        except asyncio.CancelledError:
            return
        except Exception as exc:
            get_logger().warning(f"Rollout task failed in group {meta.group_id} ({meta.env_name}): {exc!r}")
            task_idx = group.task_idx if group is not None else -1
            rollouts = [
                Rollout(task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)))
                for _ in range(meta.rollout_count)
            ]
            for r in rollouts:
                r.capture_error(exc)
            is_synth_exception = True

        self.metrics.record_completion(kind=meta.kind, env_name=meta.env_name, n=len(rollouts))
        if meta.kind == "train" and self.train_source.mixture is not None and meta.started_at:
            self.train_source.mixture.observe_completion(
                meta.env_name,
                service_seconds=time.monotonic() - meta.started_at,
                tokens_per_rollout=sum(rollout.num_total_tokens for rollout in rollouts) / len(rollouts),
            )

        for r in rollouts:
            if not r.has_error and r.num_turns == 0:
                # Empty trajectory: promote to an explicit error so the sink
                # treats it like any other failure
                r.errors.append(vf.Error(type="EmptyTrajectory", message="Rollout returned with no trajectory steps"))
                get_logger().warning(f"Empty trajectory in group {meta.group_id} ({meta.env_name})")
            if r.has_error:
                self.metrics.record_error(kind=meta.kind, env_name=meta.env_name)
                if not is_synth_exception:
                    get_logger().warning(
                        f"Rollout failed in group {meta.group_id} ({meta.env_name}) — {r.error.type}: {r.error.message}"
                    )
                can_replace = (
                    meta.kind == "train"
                    and group is not None
                    and self.max_group_replacements > 0
                    and not self.train_envs.get(meta.env_name).requires_group_scoring
                )
                if can_replace:
                    missing = group.target_rollouts - group.emitted
                    outstanding = self._group_outstanding(meta.group_id, group)
                    if outstanding >= missing:
                        continue
                    if self._request_group_replacement(group):
                        get_logger().info(
                            "Replacing failed group member | group=%s env=%s completed=%d/%d replacement=%d/%d",
                            str(meta.group_id)[:8],
                            meta.env_name,
                            group.emitted,
                            group.target_rollouts,
                            group.replacements_started,
                            self.max_group_replacements,
                        )
                        continue
                    await self._abandon_group(meta, group, r)
                    return
            await self.emit_rollout(meta, group, r)

    def _group_outstanding(self, group_id: uuid.UUID, group: GroupState) -> int:
        return group.rollouts_to_schedule + sum(
            item.rollout_count for item in self.inflight.values() if item.group_id == group_id
        )

    async def _cancel_group_remainder(self, group_id: uuid.UUID) -> int:
        claimed: list[tuple[asyncio.Task, InflightRollout]] = []
        for task, meta in list(self.inflight.items()):
            if meta.group_id != group_id:
                continue
            del self.inflight[task]
            self.release(meta.rollout_count)
            self.metrics.record_cancellation(kind=meta.kind, env_name=meta.env_name, n=meta.rollout_count)
            claimed.append((task, meta))
        if claimed:
            await safe_cancel_all([task for task, _ in claimed])
        return sum(meta.rollout_count for _, meta in claimed)

    async def _abandon_group(self, meta: InflightRollout, group: GroupState, failure: Rollout) -> None:
        self.groups.pop(meta.group_id, None)
        await self._cancel_group_remainder(meta.group_id)
        missing = group.target_rollouts - group.emitted
        for index in range(missing):
            rollout = failure
            if index > 0:
                rollout = Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=group.task_idx, prompt=None)),
                    errors=[vf.Error(type="ReplacementLimit", message="group replacement limit exhausted")],
                    stop_condition="error",
                )
            await self.emit_rollout(meta, group, rollout)

    async def emit_rollout(self, meta: InflightRollout, group: GroupState | None, rollout: Rollout) -> None:
        """Stamp prime-rl metadata onto the completed rollout and put it on ``out_q``.
        Pops the group from ``self.groups`` once every member has been emitted."""
        eval_step = meta.eval_step
        policy_version = meta.policy_version
        if group is not None:
            eval_step = group.eval_step
            policy_version = group.policy_version_at_start
            group.emitted += 1
            if group.emitted >= group.target_rollouts:
                self.groups.pop(meta.group_id, None)
                await self._cancel_group_remainder(meta.group_id)

        rollout.kind = meta.kind
        rollout.env_name = meta.env_name
        rollout.group_id = meta.group_id
        rollout.policy_version = policy_version
        rollout.off_policy_steps = meta.off_policy_steps
        if meta.kind == "eval":
            assert eval_step is not None, "eval rollout missing eval_step"
            rollout.eval_step = eval_step
        await self.out_q.put(rollout)

    async def drop_group(self, group_id: uuid.UUID) -> int:
        """Cancel remaining in-flight tasks for this group and emit a
        ``Cancelled`` marker for every rollout it still owes the sink
        (both in-flight and not-yet-scheduled). Returns the count for
        off-policy metrics."""
        group = self.groups.pop(group_id, None)
        task_idx = group.task_idx if group is not None else -1

        # Sync claim phase: pop matching tasks from ``self.inflight`` and
        # release their permits in one non-yielding sweep. After this loop
        # the dropped tasks are no longer reachable from ``self.inflight``,
        # so ``handle_completed_rollout``'s existing None-guard makes the
        # subsequent async emit phase race-free.
        claimed: list[tuple[asyncio.Task, InflightRollout]] = []
        for task, meta in list(self.inflight.items()):
            if meta.group_id != group_id:
                continue
            del self.inflight[task]
            self.release(meta.rollout_count)
            claimed.append((task, meta))

        tasks_to_cancel = [task for task, _ in claimed]
        inflight_cancelled = sum(meta.rollout_count for _, meta in claimed)
        last_meta: InflightRollout | None = claimed[-1][1] if claimed else None
        for _, meta in claimed:
            for _ in range(meta.rollout_count):
                trace = Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)),
                    errors=[vf.Error(type="Cancelled", message="Off-policy cancel")],
                    stop_condition="error",
                )
                await self.emit_rollout(meta, group, trace)

        # For non-group-scoring envs, the group may have rollouts that
        # were never dispatched (``rollouts_to_schedule > 0``). Emit
        # markers for those too so the sink hits ``target_rollouts``
        #
        # ``last_meta`` can be ``None`` if the only inflight task for this
        # group completed naturally between ``on_version_pending``'s snapshot
        # and us reaching it — synthesize a stand-in from the group state
        unscheduled_cancelled = 0
        if group is not None and group.rollouts_to_schedule > 0:
            fallback_meta = last_meta or InflightRollout(
                kind=group.kind,
                env_name=group.env_name,
                group_id=group_id,
                policy_version=group.policy_version_at_start,
                rollout_count=1,
                eval_step=group.eval_step,
            )
            unscheduled_cancelled = group.rollouts_to_schedule
            for _ in range(unscheduled_cancelled):
                trace = Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)),
                    errors=[vf.Error(type="Cancelled", message="Off-policy cancel")],
                    stop_condition="error",
                )
                await self.emit_rollout(fallback_meta, group, trace)

        cancelled = inflight_cancelled + unscheduled_cancelled
        if cancelled > 0:
            meta_for_log = last_meta or (
                InflightRollout(
                    kind=group.kind,
                    env_name=group.env_name,
                    group_id=group_id,
                    policy_version=group.policy_version_at_start if group else 0,
                    rollout_count=1,
                    eval_step=group.eval_step,
                )
                if group is not None
                else None
            )
            if meta_for_log is not None:
                self.metrics.record_cancellation(kind=meta_for_log.kind, env_name=meta_for_log.env_name, n=cancelled)
                get_logger().debug(
                    f"drain {meta_for_log.kind} | group={str(group_id)[:8]} env={meta_for_log.env_name} | "
                    f"cancelled={cancelled} (inflight={inflight_cancelled} unscheduled={unscheduled_cancelled})"
                )

        if tasks_to_cancel:
            await safe_cancel_all(tasks_to_cancel)
        return cancelled

    async def cancel_inflight_rollouts(self) -> None:
        """Cancel all in-flight rollouts. Used on shutdown — doesn't emit
        markers since the sinks are being torn down anyway."""
        for meta in self.inflight.values():
            self.metrics.record_cancellation(kind=meta.kind, env_name=meta.env_name, n=meta.rollout_count)
            self.release(meta.rollout_count)
        tasks = list(self.inflight.keys())
        self.inflight.clear()
        self.groups.clear()
        if tasks:
            await safe_cancel_all(tasks)

    async def cancel_inflight_train_rollouts(self) -> int:
        """Cancel in-flight train rollouts, leaving eval alone. Used by the
        orchestrator at ``max_steps`` so triggered eval can still complete
        through the pipeline while wasted train inference is short-circuited."""
        train_tasks: list[asyncio.Task] = []
        train_group_ids: set[uuid.UUID] = set()
        cancelled = 0
        for task, meta in list(self.inflight.items()):
            if meta.kind != "train":
                continue
            self.inflight.pop(task, None)
            self.release(meta.rollout_count)
            self.metrics.record_cancellation(kind="train", env_name=meta.env_name, n=meta.rollout_count)
            cancelled += meta.rollout_count
            train_tasks.append(task)
            train_group_ids.add(meta.group_id)
        for gid in train_group_ids:
            self.groups.pop(gid, None)
        if train_tasks:
            await safe_cancel_all(train_tasks)
        return cancelled

    # ── metrics ────────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        """Instantaneous, read-only gauges sampled by the periodic logger."""
        return {
            "dispatcher/inflight_train": float(self.inflight_train_count),
            "dispatcher/inflight_eval": float(self.inflight_eval_count),
            "dispatcher/queued/eval": float(self.queued_eval_examples),
            "dispatcher/mode": float(self.mode == DispatcherMode.PREFER_EVAL),
            "dispatcher/groups_in_flight": float(len(self.groups)),
            "dispatcher/off_policy_level_max": float(self.max_off_policy_level),
            "dispatcher/off_policy_level_mean": self.mean_off_policy_level,
        }
