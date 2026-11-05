"""Difficulty-pool task sampling."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.samplers.base import TaskSampler

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import DifficultyPoolSamplerConfig


def valid_rewards(group: list[vf.Episode]) -> list[float]:
    """Rewards of a group's clean trainable traces."""
    return [
        trace.reward
        for episode in group
        for trace in episode.traces
        if not trace.has_error and trace.agent.trainable
    ]


class DifficultyPoolSampler(TaskSampler):
    """Weight finite tasks by a pool derived from their latest group mean.

    Each pool's threshold is its inclusive maximum reward; the final pool is
    the catch-all. Unseen tasks have neutral weight, so observations affect
    sampling immediately without requiring a full taskset pass.
    """

    def __init__(
        self,
        config: DifficultyPoolSamplerConfig,
        tasks: Sequence[vf.Task] | Iterator[vf.Task],
        *,
        task_key_cache: dict[int, str] | None = None,
    ) -> None:
        if not isinstance(tasks, Sequence):
            raise ValueError("DifficultyPoolSampler requires a finite taskset")
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise ValueError("DifficultyPoolSampler requires at least one task")
        cache = {} if task_key_cache is None else task_key_cache
        keys = []
        for task in self.tasks:
            identity = id(task)
            if identity not in cache:
                cache[identity] = task.key
            keys.append(cache[identity])
        duplicates = {key for key, count in Counter(keys).items() if count > 1}
        if duplicates:
            raise ValueError(f"Task keys must be unique within a taskset: {sorted(duplicates)}")
        self.task_keys = tuple(keys)
        self.tasks_by_key = dict(zip(keys, self.tasks))
        self.rng = random.Random(config.seed)
        self.pools = config.pools
        self.strategy = config.strategy
        self.shared_group = config.shared_group
        self.stagger = config.stagger_harnesses
        # Unseen tasks a staggered peer is exploring; TrainSource gives all peers one set.
        self.claimed: set[str] = set()
        # Tasks whose first group on a peer harness came out mixed, explored here next.
        self.offered: deque[str] = deque()
        self.history_size = config.history_size
        self.retire_solved_after = config.retire_solved_after
        self.retire_solved_rate = config.retire_solved_rate
        self.retire_unsolved_after = config.retire_unsolved_after
        self.retired: set[str] = set()
        self.replay_fraction = config.replay_fraction
        self.replay_warmup_tasks = config.replay_warmup_tasks
        self.task_history: dict[str, list[float]] = {}
        ordered = sorted(self.pools.items(), key=lambda item: item[1].threshold)
        self._ordered_pools = tuple(ordered)
        self.task_rewards: dict[str, float] = {}
        self.excluded_keys: set[str] = set()
        # Smoothed replay explores unseen tasks in one seeded order, so the next
        # unseen tasks are known ahead of sampling (see ``upcoming``). Samplers that
        # share outcomes also share the order: one harness's scored task is skipped
        # by the others, and each new task needs one image build for all of them.
        self._explore_order = list(self.task_keys)
        explore_seed = config.shared_group if config.shared_group is not None else config.seed
        random.Random(f"{explore_seed}:explore").shuffle(self._explore_order)
        self._explore_cursor = 0
        self._explore_digest = hashlib.sha256("\n".join(self._explore_order).encode()).hexdigest()

    def task_pool(self, task_key: str) -> str | None:
        """Return the task's current pool, or ``None`` until it has a score."""
        score = self.task_rewards.get(task_key)
        if score is None:
            return None
        for name, pool in self._ordered_pools:
            if score <= pool.threshold:
                return name
        return self._ordered_pools[-1][0]

    @property
    def has_available_tasks(self) -> bool:
        return len(self.excluded_keys) < len(self.tasks)

    def exclude_tasks(self, task_keys: set[str]) -> None:
        self.excluded_keys.update(task_keys & self.tasks_by_key.keys())

    def __next__(self) -> vf.Task:
        if not self.has_available_tasks:
            raise StopIteration
        if self.strategy == "smoothed_replay":
            while self.offered:
                key = self.offered.popleft()
                if key not in self.excluded_keys:
                    return self.tasks_by_key[key]
            available = [key for key in self.task_keys if key not in self.excluded_keys]
            observed = [key for key in available if key in self.task_rewards]
            unseen = [key for key in available if key not in self.task_rewards]
            replay = self.replay_fraction * min(1.0, len(observed) / self.replay_warmup_tasks)
            if observed and self.rng.random() < replay:
                weights = [max(0.05, 4 * self.task_rewards[key] * (1 - self.task_rewards[key]))
                           for key in observed]
                key = self.rng.choices(observed, weights=weights, k=1)[0]
            else:
                key = self._next_unexplored() or self.rng.choice(unseen or available)
                if self.stagger:
                    self.claimed.add(key)
            return self.tasks_by_key[key]
        weights = []
        for task_key in self.task_keys:
            pool = self.task_pool(task_key)
            weights.append(0.0 if task_key in self.excluded_keys else (1.0 if pool is None else self.pools[pool].weight))
        if not any(weights):
            raise RuntimeError("DifficultyPoolSampler has no tasks in a pool with positive weight")
        return self.rng.choices(self.tasks, weights=weights, k=1)[0]

    def _unexplored(self, key: str) -> bool:
        return key not in self.excluded_keys and key not in self.task_rewards and key not in self.claimed

    def _next_unexplored(self) -> str | None:
        while self._explore_cursor < len(self._explore_order):
            key = self._explore_order[self._explore_cursor]
            self._explore_cursor += 1
            if self._unexplored(key):
                return key
        return None

    def upcoming(self, count: int) -> list[vf.Task]:
        """The next unseen tasks smoothed replay will explore, without sampling them.

        Replay draws among observed tasks stay random; once every task has been
        explored, unseen draws fall back to a random choice and nothing is known ahead.
        """
        if self.strategy != "smoothed_replay":
            return []
        found = []
        for key in self._explore_order[self._explore_cursor:]:
            if len(found) == count:
                break
            if self._unexplored(key):
                found.append(self.tasks_by_key[key])
        return found

    def offer(self, key: str) -> None:
        """Explore a task next that a staggered peer harness found mixed."""
        if key in self.tasks_by_key:
            self.offered.append(key)

    def observe(self, group: list[vf.Episode]) -> None:
        rewards = valid_rewards(group)
        if not rewards:
            return
        task = group[0].task
        if task.key is None:
            raise ValueError("A finalized group is missing Task.key")
        if task.key not in self.tasks_by_key:
            return
        if self.strategy == "smoothed_replay":
            history = (self.task_history.get(task.key, []) + rewards)[-self.history_size:]
            self.task_history[task.key] = history
            # Beta(1, 1) smoothing: 0/8 is uncertain, while 2/8 remains useful.
            self.task_rewards[task.key] = (1.0 + sum(history)) / (2.0 + len(history))
            self._maybe_retire(task.key)
        else:
            self.task_rewards[task.key] = sum(rewards) / len(rewards)

    def _maybe_retire(self, key: str) -> None:
        history = self.task_history[key]
        solved = (self.retire_solved_after is not None and len(history) >= self.retire_solved_after
                  and self.task_rewards[key] >= self.retire_solved_rate)
        unsolved = (self.retire_unsolved_after is not None and len(history) >= self.retire_unsolved_after
                    and not any(history))
        if solved or unsolved:
            self.retired.add(key)
            self.excluded_keys.add(key)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "explore_cursor": self._explore_cursor,
            "explore_digest": self._explore_digest,
            "task_rewards": dict(self.task_rewards),
            **({"task_history": {key: list(values) for key, values in self.task_history.items()}}
               if self.strategy == "smoothed_replay" else {}),
            # Claims are not saved: their groups are in flight and die with the run.
            **({"offered": list(self.offered)} if self.stagger else {}),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng.setstate(state_dict["rng"])
        # The cursor indexes this taskset's explore order; a changed task list reorders it, so start over
        # (observed tasks are skipped). A checkpoint from before the digest was saved restarts too.
        same_order = state_dict.get("explore_digest") == self._explore_digest
        self._explore_cursor = state_dict.get("explore_cursor", 0) if same_order else 0
        self.offered = deque(key for key in state_dict.get("offered", []) if key in self.tasks_by_key)
        self.task_rewards = dict(state_dict["task_rewards"])
        if self.strategy == "smoothed_replay":
            histories = state_dict.get("task_history")
            if histories is None:
                # A legacy mean is one weak observation, not eight fabricated trials.
                histories = {key: [value] for key, value in self.task_rewards.items()}
            self.task_history = {key: list(values)[-self.history_size:] for key, values in histories.items()}
            self.task_rewards = {key: (1.0 + sum(values)) / (2.0 + len(values))
                                 for key, values in self.task_history.items()}
            for key in self.task_history:
                if key in self.tasks_by_key:
                    self._maybe_retire(key)

    def metrics(self) -> dict[str, float]:
        occupancy = dict.fromkeys(self.pools, 0)
        for task_key in self.task_rewards:
            pool = self.task_pool(task_key)
            if pool is not None:
                occupancy[pool] += 1
        return {
            "pool/unseen": float(sum(key not in self.task_rewards for key in self.task_keys)),
            **({"pool/retired": float(len(self.retired))}
               if self.retire_solved_after is not None or self.retire_unsolved_after is not None else {}),
            **{f"pool/{name}": float(count) for name, count in occupancy.items()},
        }
