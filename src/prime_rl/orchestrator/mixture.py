"""Bounded rollout-mixture planning for multi-environment training."""

from __future__ import annotations

import math
from collections.abc import Mapping


class MixturePlanner:
    """Plan whole-group batches and bound queued plus in-flight supply."""

    def __init__(
        self,
        *,
        ratios: Mapping[str, float],
        group_sizes: Mapping[str, int],
        batch_size: int,
        max_inflight: int,
    ) -> None:
        total_ratio = sum(ratios.values())
        self.env_names = list(ratios)
        self.weights = {name: ratios[name] / total_ratio for name in self.env_names}
        self.group_sizes = dict(group_sizes)
        self.batch_size = batch_size
        self.credits = {name: 0.0 for name in self.env_names}
        self.inventory_limits = {
            name: max(self.group_sizes[name], math.ceil(max_inflight * self.weights[name])) for name in self.env_names
        }

    def plan_batch(self) -> tuple[str, ...]:
        """Return an environment plan whose whole groups fill one batch."""
        for name, weight in self.weights.items():
            self.credits[name] += weight * self.batch_size

        plan: list[str] = []
        planned_rollouts = 0
        while planned_rollouts < self.batch_size:
            name = max(self.env_names, key=self.credits.__getitem__)
            plan.append(name)
            group_size = self.group_sizes[name]
            self.credits[name] -= group_size
            planned_rollouts += group_size
        return tuple(plan)

    def environments_needing_work(
        self,
        *,
        pending: Mapping[str, int],
        inflight: Mapping[str, int],
        costs: Mapping[str, int],
        available_permits: int,
    ) -> list[str]:
        """Return environments whose bounded future inventory has room."""
        return [
            name
            for name in self.env_names
            if costs[name] <= available_permits
            and pending.get(name, 0) + inflight.get(name, 0) + costs[name] <= self.inventory_limits[name]
        ]
