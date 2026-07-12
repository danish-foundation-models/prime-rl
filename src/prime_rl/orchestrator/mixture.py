"""Bounded rollout-mixture planning for multi-environment training."""

from __future__ import annotations

import math
from collections.abc import Mapping


class MixturePlanner:
    """Plan optimizer mixtures and steer bounded rollout supply from observed service time."""

    def __init__(
        self,
        *,
        ratios: Mapping[str, float],
        group_sizes: Mapping[str, int],
        batch_size: int,
        max_inflight: int,
        service_time_alpha: float | None = None,
        success_rate_alpha: float | None = None,
        max_supply_multiplier: float = 1.0,
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
        self.supply_limits = {
            name: math.ceil(max_supply_multiplier * self.inventory_limits[name]) for name in self.env_names
        }
        self.max_inflight = max_inflight
        self.service_time_alpha = service_time_alpha
        self.success_rate_alpha = success_rate_alpha
        self._service_seconds: dict[str, float | None] = {name: None for name in self.env_names}
        self._tokens_per_rollout: dict[str, float | None] = {name: None for name in self.env_names}
        self._success_rate = {name: 1.0 for name in self.env_names}

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

    def observe_completion(
        self,
        name: str,
        *,
        service_seconds: float,
        tokens_per_rollout: float,
        success: bool = True,
    ) -> None:
        """Update delayed rollout-cost estimates without treating unseen strata as cheap."""
        if self.service_time_alpha is not None:
            previous_service = self._service_seconds[name]
            self._service_seconds[name] = (
                service_seconds
                if previous_service is None
                else self.service_time_alpha * service_seconds + (1 - self.service_time_alpha) * previous_service
            )
            previous_tokens = self._tokens_per_rollout[name]
            self._tokens_per_rollout[name] = (
                tokens_per_rollout
                if previous_tokens is None
                else self.service_time_alpha * tokens_per_rollout + (1 - self.service_time_alpha) * previous_tokens
            )
        if self.success_rate_alpha is not None:
            observed = float(success)
            previous = self._success_rate[name]
            self._success_rate[name] = self.success_rate_alpha * observed + (1 - self.success_rate_alpha) * previous

    def _estimates(self, values: Mapping[str, float | None], default: float) -> dict[str, float]:
        observed = [value for value in values.values() if value is not None]
        fallback = max(observed, default=default)
        return {name: value if value is not None else fallback for name, value in values.items()}

    @property
    def service_seconds(self) -> dict[str, float]:
        return self._estimates(self._service_seconds, 1.0)

    @property
    def tokens_per_rollout(self) -> dict[str, float]:
        return self._estimates(self._tokens_per_rollout, 0.0)

    @property
    def success_rate(self) -> dict[str, float]:
        return dict(self._success_rate)

    @property
    def effective_supply_limits(self) -> dict[str, int]:
        if self.success_rate_alpha is None:
            return dict(self.supply_limits)
        limits = {}
        for name in self.env_names:
            baseline = self.inventory_limits[name]
            group_size = self.group_sizes[name]
            extra_groups = math.floor((self.supply_limits[name] - baseline) * self._success_rate[name] / group_size)
            limits[name] = baseline + extra_groups * group_size
        return limits

    def environments_needing_work(
        self,
        *,
        pending: Mapping[str, int],
        inflight: Mapping[str, int],
        costs: Mapping[str, int],
        available_permits: int,
    ) -> list[str]:
        """Return environments with room in their configured future-supply bound."""
        supply_limits = self.effective_supply_limits
        return [
            name
            for name in self.env_names
            if costs[name] <= available_permits
            and pending.get(name, 0) < self.inventory_limits[name]
            and pending.get(name, 0) + inflight.get(name, 0) + costs[name] <= supply_limits[name]
        ]

    def choose_environment(self, *, eligible: list[str], inflight: Mapping[str, int]) -> str:
        """Choose the largest service-time-weighted concurrency deficit."""
        if self.service_time_alpha is None:
            raise ValueError("service-time feedback is not configured")

        service = self.service_seconds
        demand = {name: self.weights[name] * service[name] for name in eligible}
        total_demand = sum(demand.values())
        targets = {name: self.max_inflight * demand[name] / total_demand for name in eligible}
        return max(
            eligible,
            key=lambda name: (targets[name] - inflight.get(name, 0)) / max(targets[name], self.group_sizes[name]),
        )
