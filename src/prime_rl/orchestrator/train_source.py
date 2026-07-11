"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights are each env's configured ``ratio`` (default 1, i.e. equal weight
per env). ``next_example`` reshuffles on cursor exhaustion."""

from __future__ import annotations

import random
from collections.abc import Mapping

from prime_rl.orchestrator.envs import TrainEnvs


class TrainSource:
    """``next_example(available_permits)`` picks a weighted-RR env and
    returns its next example (or ``None`` when no environment fits the global
    permits and per-environment in-flight limits).
    Returned dicts carry ``env_name`` + ``task_idx``."""

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        self.examples: dict[str, list[dict]] = {}
        self.cursors: dict[str, int] = {}
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
            self.env_costs[env.name] = env.config.group_size if env.requires_group_scoring else 1

        self.env_names = [e.name for e in self.envs]
        self.weights: dict[str, float] = {e.name: float(e.config.ratio) for e in self.envs}
        self.max_inflight: dict[str, int | None] = {e.name: e.config.max_inflight for e in self.envs}

    def next_example(self, available_permits: int, inflight_by_env: Mapping[str, int]) -> dict | None:
        eligible = [
            env_name
            for env_name in self.env_names
            if self.env_costs[env_name] <= available_permits
            and (
                self.max_inflight[env_name] is None
                or inflight_by_env.get(env_name, 0) + self.env_costs[env_name] <= self.max_inflight[env_name]
            )
        ]
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
