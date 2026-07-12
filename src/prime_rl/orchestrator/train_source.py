"""TrainSource: inventory-aware sampling across train environments.

Rollout-batched runs stop requesting an environment when its queued and in-flight
inventory fills its mixture share. ``next_example`` reshuffles on cursor exhaustion."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping

from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.mixture import MixturePlanner


class TrainSource:
    """Pick an environment that needs future supply and return its next example.

    Returns ``None`` when no environment fits the available permits or all bounded
    inventories are full. Returned dicts carry ``env_name`` and ``task_idx``.
    """

    def __init__(
        self,
        train_envs: TrainEnvs,
        *,
        seed: int | None,
        mixture: MixturePlanner | None = None,
        pending_by_env: Callable[[], Mapping[str, int]] | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        self.examples: dict[str, list[dict]] = {}
        self.cursors: dict[str, int] = {}
        self.group_sizes: dict[str, int] = {}
        # Group-scoring envs reserve ``group_size`` permits up front;
        # per-rollout envs need 1
        self.env_costs: dict[str, int] = {}
        for env in self.envs:
            # The orchestrator never loads the env: sample over the task-index
            # range the server reported via info() (num_tasks).
            rows: list[dict] = [{"task_idx": i, "env_name": env.name} for i in range(env.num_tasks)]
            self.rng.shuffle(rows)
            self.examples[env.name] = rows
            self.cursors[env.name] = 0
            self.group_sizes[env.name] = env.config.group_size
            self.env_costs[env.name] = env.config.group_size if env.requires_group_scoring else 1

        self.env_names = [e.name for e in self.envs]
        self.weights: dict[str, float] = {e.name: float(e.config.ratio) for e in self.envs}
        self.mixture = mixture
        self.pending_by_env = pending_by_env

    def next_example(self, available_permits: int, inflight_by_env: Mapping[str, int] | None = None) -> dict | None:
        if self.mixture is None:
            eligible = [name for name in self.env_names if self.env_costs[name] <= available_permits]
        else:
            assert self.pending_by_env is not None
            eligible = self.mixture.environments_needing_work(
                pending=self.pending_by_env(),
                inflight=inflight_by_env or {},
                costs=self.group_sizes,
                available_permits=available_permits,
            )
        if not eligible:
            return None
        env_name = self.rng.choices(eligible, weights=[self.weights[name] for name in eligible], k=1)[0]
        rows = self.examples[env_name]
        cursor = self.cursors[env_name]
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
