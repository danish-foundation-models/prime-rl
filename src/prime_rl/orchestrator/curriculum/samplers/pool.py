"""Difficulty-pool task sampling."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.samplers.base import TaskSampler

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import DifficultyPoolSamplerConfig


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
        self.history_size = config.history_size
        self.replay_fraction = config.replay_fraction
        self.replay_warmup_tasks = config.replay_warmup_tasks
        self.task_history: dict[str, list[float]] = {}
        ordered = sorted(self.pools.items(), key=lambda item: item[1].threshold)
        self._ordered_pools = tuple(ordered)
        self.task_rewards: dict[str, float] = {}
        self.excluded_keys: set[str] = set()

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
            available = [key for key in self.task_keys if key not in self.excluded_keys]
            observed = [key for key in available if key in self.task_rewards]
            unseen = [key for key in available if key not in self.task_rewards]
            replay = self.replay_fraction * min(1.0, len(observed) / self.replay_warmup_tasks)
            if observed and self.rng.random() < replay:
                weights = [max(0.05, 4 * self.task_rewards[key] * (1 - self.task_rewards[key]))
                           for key in observed]
                key = self.rng.choices(observed, weights=weights, k=1)[0]
            else:
                key = self.rng.choice(unseen or available)
            return self.tasks_by_key[key]
        weights = []
        for task_key in self.task_keys:
            pool = self.task_pool(task_key)
            weights.append(0.0 if task_key in self.excluded_keys else (1.0 if pool is None else self.pools[pool].weight))
        if not any(weights):
            raise RuntimeError("DifficultyPoolSampler has no tasks in a pool with positive weight")
        return self.rng.choices(self.tasks, weights=weights, k=1)[0]

    def observe(self, group: list[vf.Episode]) -> None:
        rewards = [
            trace.reward
            for episode in group
            for trace in episode.traces
            if not trace.has_error and trace.agent.trainable
        ]
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
        else:
            self.task_rewards[task.key] = sum(rewards) / len(rewards)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "task_rewards": dict(self.task_rewards),
            **({"task_history": {key: list(values) for key, values in self.task_history.items()}}
               if self.strategy == "smoothed_replay" else {}),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng.setstate(state_dict["rng"])
        self.task_rewards = dict(state_dict["task_rewards"])
        if self.strategy == "smoothed_replay":
            histories = state_dict.get("task_history")
            if histories is None:
                # A legacy mean is one weak observation, not eight fabricated trials.
                histories = {key: [value] for key, value in self.task_rewards.items()}
            self.task_history = {key: list(values)[-self.history_size:] for key, values in histories.items()}
            self.task_rewards = {key: (1.0 + sum(values)) / (2.0 + len(values))
                                 for key, values in self.task_history.items()}

    def metrics(self) -> dict[str, float]:
        occupancy = dict.fromkeys(self.pools, 0)
        for task_key in self.task_rewards:
            pool = self.task_pool(task_key)
            if pool is not None:
                occupancy[pool] += 1
        return {
            "pool/unseen": float(len(self.tasks_by_key) - len(self.task_rewards)),
            **{f"pool/{name}": float(count) for name, count in occupancy.items()},
        }
