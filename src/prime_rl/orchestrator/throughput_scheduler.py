"""Separate rollout inventory from balanced consumption of completed traces."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque


def allocate_group_budgets(demand, duration, mean_oversampling, minimum, maximum):
    """MiMo eq. 6: solve the demand-weighted clipped oversampling constraint."""
    if not 0 <= minimum <= mean_oversampling <= maximum:
        raise ValueError("Require 0 <= minimum <= mean oversampling <= maximum")
    if not demand or any(m <= 0 or not math.isfinite(m) for m in demand.values()):
        raise ValueError("Demand must be finite and positive")
    if any(duration[n] <= 0 or not math.isfinite(duration[n]) for n in demand):
        raise ValueError("Duration must be finite and positive")
    total = sum(demand.values())
    lo, hi = 0.0, (1 + maximum) / min(duration[n] for n in demand)
    for _ in range(60):
        scale = (lo + hi) / 2
        mean = sum(m * min(maximum, max(minimum, scale * duration[n] - 1)) for n, m in demand.items()) / total
        if mean < mean_oversampling:
            lo = scale
        else:
            hi = scale
    scale = (lo + hi) / 2
    return {n: m * (1 + min(maximum, max(minimum, scale * duration[n] - 1))) for n, m in demand.items()}


class ThroughputScheduler:
    def __init__(
        self,
        sources: list[tuple[str, str, float, int]],
        window: int,
        *,
        category_targets: dict[str, float] | None = None,
        batch_tolerance: int = 2,
        max_surplus: int = 8,
        batch_size: int | None = None,
        target_mix: float = 0.5,
        oversampling: float = 3.0,
        min_oversampling: float = 0.0,
        max_oversampling: float = 16.0,
        acceptance_floor: float = 0.05,
        inventory_factor: float = 2.0,
    ):
        self.sources = {name: (family, weight, group) for name, family, weight, group in sources}
        if len(self.sources) != len(sources) or any(
            not math.isfinite(weight) or weight < 0 or group < 1 for _, _, weight, group in sources
        ):
            raise ValueError("Unique sources with finite nonnegative weights and positive group sizes required")
        total = sum(weight for _, _, weight, _ in sources)
        if total <= 0 or window <= 0:
            raise ValueError("Positive source weight and inventory window required")
        self.weights = {name: weight / total for name, (_, weight, _) in self.sources.items()}
        self.families = {name: family for name, (family, _, _) in self.sources.items()}
        self.family_weights = defaultdict(float)
        for name, weight in self.weights.items():
            self.family_weights[self.families[name]] += weight
        self.category_targets = dict(category_targets or {})
        self.batch_tolerance = batch_tolerance
        self.max_surplus = max_surplus
        self.batch_size = batch_size
        if not 0 <= target_mix <= 1 or not 0 < acceptance_floor <= 1 or not inventory_factor >= 1:
            raise ValueError("Invalid MiMo scheduling mix, acceptance floor, or inventory factor")
        if not 0 <= min_oversampling <= oversampling <= max_oversampling < float("inf"):
            raise ValueError("Invalid MiMo oversampling bounds")
        self.target_mix = target_mix
        self.oversampling = oversampling
        self.min_oversampling = min_oversampling
        self.max_oversampling = max_oversampling
        self.acceptance_floor = acceptance_floor
        self.inventory_factor = inventory_factor
        self.round_robin_credit = dict.fromkeys(self.sources, 0.0)
        self.startup_dispatched = 0
        self.last_group_budgets = {}
        self.last_scheduling_weights = {}
        self.last_mean_oversampling = oversampling
        if self.category_targets:
            if set(self.category_targets) != set(self.family_weights):
                raise ValueError("category_targets must cover exactly the configured source categories")
            if batch_size is None or batch_size < 1 or batch_tolerance < 1 or max_surplus < 1:
                raise ValueError("Category balance requires positive batch size, tolerance and surplus")
            if any(not math.isfinite(w) or w <= 0 for w in self.category_targets.values()):
                raise ValueError("Category targets must be finite and positive")
            target_total = sum(self.category_targets.values())
            self.category_targets = {f: w / target_total for f, w in self.category_targets.items()}
            self.weights = {
                n: w / self.family_weights[self.families[n]] * self.category_targets[self.families[n]]
                for n, w in self.weights.items()
            }
            self.family_weights = self.category_targets.copy()
        self.scopes = {n: self.families[n] if self.category_targets else n for n in self.sources}
        self.scope_round_robin_credit = dict.fromkeys(set(self.scopes.values()), 0.0)
        self.scope_group_sizes = {}
        for name, scope in self.scopes.items():
            size = self.sources[name][2]
            if scope in self.scope_group_sizes and self.scope_group_sizes[scope] != size:
                raise ValueError("MiMo scopes require a common GRPO group size within each category")
            self.scope_group_sizes[scope] = size
        self.batch_shortfall = {}
        self.batch_balance_blocked = False
        self.last_batch_counts = {}
        self.window = window
        self.served = Counter()
        self.family_served = Counter()
        self.total_served = 0
        self.duration = {name: 600.0 for name in self.sources}
        self.yield_rate = dict.fromkeys(self.sources, 1.0)
        self.last_block = "none"
        self.last_batch_share_error = 0.0

    def observe_duration(self, name: str, duration: float):
        if math.isfinite(duration) and duration > 0:
            self.duration[name] = 0.8 * self.duration[name] + 0.2 * duration

    def observe_yield(self, name: str, attempted: int, usable: int):
        if attempted:
            self.yield_rate[name] = 0.9 * self.yield_rate[name] + 0.1 * min(1.0, usable / attempted)

    def _debt(self, key: str, weights, counts, total: int) -> float:
        # Bound catch-up after unavailable sources; never demand an unlimited debt repayment.
        bound = max(8.0, self.window * weights[key])
        return max(-bound, min(bound, total * weights[key] - counts[key]))

    def choose_sources(
        self,
        ready: dict[str, int],
        outstanding: dict[str, int],
        capacity: int,
        *,
        buffered_tokens: int = 0,
        token_limit: int | None = None,
    ) -> set[str]:
        self.last_block = "none"
        if not self.category_targets and sum(ready.values()) >= self.window:
            self.last_block = "ready_inventory"
            return set()
        if token_limit is not None and buffered_tokens >= token_limit:
            self.last_block = "buffered_tokens"
            return set()
        # A group with seven completed members must not consume eight active slots.
        # Bound total unretired work separately from actual concurrency.
        max_group = max(v[2] for v in self.sources.values())
        if sum(outstanding.values()) + sum(ready.values()) + max_group > self.window + max_group * capacity:
            self.last_block = "unfinished_inventory"
            return set()
        family_ready = Counter()
        for name, count in ready.items():
            family_ready[self.families[name]] += count
        candidates = [
            name
            for name, weight in self.weights.items()
            if weight > 0
            and (
                not self.category_targets
                or self.family_served[self.families[name]]
                < (self.total_served + self.batch_size) * self.family_weights[self.families[name]] + self.max_surplus
            )
            and family_ready[self.families[name]]
            < max(self.sources[name][2], math.ceil(self.window * self.family_weights[self.families[name]]))
        ]
        if not candidates:
            self.last_block = "family_ready_inventory"
            return set()
        budgets, scheduling_weights = self.scheduling_plan(ready, capacity)
        outstanding_groups = Counter()
        for name, count in outstanding.items():
            outstanding_groups[self.scopes[name]] += count / self.sources[name][2]
        candidates = [
            n
            for n in candidates
            if outstanding_groups[self.scopes[n]] + 1 <= max(1, math.ceil(budgets[self.scopes[n]] - 1e-9))
        ]
        if not candidates:
            self.last_block = "source_group_budget"
            return set()
        eligible_scopes = {self.scopes[n] for n in candidates}
        # Initial occupancy is proportional to duration * generation demand.
        if self.startup_dispatched < capacity:
            scheduling_weights = {
                scope: self.scope_duration[scope] * self.scope_demand[scope] for scope in eligible_scopes
            }
        else:
            scheduling_weights = {scope: scheduling_weights[scope] for scope in eligible_scopes}
        if sum(scheduling_weights.values()) <= 0:
            self.last_block = "no_generation_demand"
            return set()
        total_weight = sum(scheduling_weights.values())
        self.last_scheduling_weights = {
            scope: scheduling_weights.get(scope, 0) / total_weight for scope in self.scope_round_robin_credit
        }
        scope = self._smooth_pick(scheduling_weights, self.scope_round_robin_credit, self.scope_round_robin_credit)
        # Within a coarse category, keep the configured task/harness mix instead
        # of making every low-yield taskset an independent hard target.
        members = {n: self.weights[n] / self.sources[n][2] for n in candidates if self.scopes[n] == scope}
        best = self._smooth_pick(members, self.round_robin_credit, [n for n in self.sources if self.scopes[n] == scope])
        self.startup_dispatched += self.sources[best][2]
        return {best}

    @staticmethod
    def _smooth_pick(weights, credits, universe):
        total = sum(weights.values())
        for name in universe:
            if name in weights:
                credits[name] += weights[name] / total
            else:
                credits[name] = 0.0
        best = max(sorted(weights), key=lambda n: credits[n])
        credits[best] -= 1.0
        return best

    def scheduling_plan(self, ready: dict[str, int], capacity: int):
        """MiMo eqs. 6/7 at target granularity: categories, or individual sources without categories."""
        batch_size = self.batch_size or self.window / (1 + self.oversampling)
        target_by_source = {n: batch_size * w / self.sources[n][2] for n, w in self.weights.items() if w > 0}
        targets = Counter()
        duration = Counter()
        acceptance = Counter()
        accepted = Counter()
        for name, target in target_by_source.items():
            scope = self.scopes[name]
            targets[scope] += target
            duration[scope] += target * self.duration[name]
            acceptance[scope] += target * self.yield_rate[name]
            accepted[scope] += ready.get(name, 0) / self.sources[name][2]
        duration = {scope: value / targets[scope] for scope, value in duration.items()}
        acceptance = {scope: max(self.acceptance_floor, value / targets[scope]) for scope, value in acceptance.items()}
        demand = {scope: target / acceptance[scope] for scope, target in targets.items()}
        nominal_episodes = sum(m * self.scope_group_sizes[scope] for scope, m in demand.items())
        mean = min(
            self.max_oversampling, max(self.oversampling, self.inventory_factor * capacity / nominal_episodes - 1)
        )
        budgets = allocate_group_budgets(demand, duration, mean, self.min_oversampling, self.max_oversampling)
        weights = {
            scope: (self.target_mix * target + (1 - self.target_mix) * max(0, target - accepted[scope]))
            / acceptance[scope]
            for scope, target in targets.items()
        }
        self.scope_duration, self.scope_demand = duration, demand
        self.last_group_budgets = budgets
        self.last_mean_oversampling = mean
        return budgets, weights

    def _category_plan(self, available: Counter, count: int) -> Counter | None:
        """Bound every batch and cumulative excess; return None rather than ship a biased batch."""
        planned = Counter()
        caps = {}
        self.batch_shortfall = {}
        for family, weight in self.family_weights.items():
            lower = max(0, math.ceil(count * weight - self.batch_tolerance - 1e-9))
            cumulative_cap = math.floor(
                (self.total_served + count) * weight + self.max_surplus - self.family_served[family] + 1e-9
            )
            upper = max(0, min(math.floor(count * weight + self.batch_tolerance + 1e-9), cumulative_cap))
            caps[family] = min(available[family], upper)
            if available[family] < lower:
                self.batch_shortfall[family] = lower - available[family]
            planned[family] = lower
        if any(caps[f] < planned[f] for f in planned):
            return None
        while sum(planned.values()) < count:
            choices = [f for f in self.family_weights if planned[f] < caps[f]]
            if not choices:
                return None
            family = max(
                sorted(choices),
                key=lambda f: (self.total_served + count) * self.family_weights[f] - self.family_served[f] - planned[f],
            )
            planned[family] += 1
        return planned

    def select(self, candidates: list[tuple[str, str]], count: int) -> list[str]:
        """Choose only finalized-group traces, FIFO within source; no reward/length filtering."""
        queues = defaultdict(deque)
        for trace_id, source in candidates:
            queues[source].append(trace_id)
        category_plan = None
        if self.category_targets:
            available = Counter()
            for name, queue in queues.items():
                available[self.families[name]] += len(queue)
            category_plan = self._category_plan(available, count)
            self.batch_balance_blocked = category_plan is None
            if category_plan is None:
                return []
        category_selected = Counter()
        served = self.served.copy()
        family_served = self.family_served.copy()
        total = self.total_served
        selected = []
        for _ in range(min(count, len(candidates))):
            families = {
                self.families[name]
                for name, queue in queues.items()
                if queue
                and (
                    category_plan is None or category_selected[self.families[name]] < category_plan[self.families[name]]
                )
            }
            family = max(sorted(families), key=lambda f: self._debt(f, self.family_weights, family_served, total + 1))
            names = [n for n, queue in queues.items() if queue and self.families[n] == family]
            name = max(names, key=lambda n: self._debt(n, self.weights, served, total + 1))
            selected.append(queues[name].popleft())
            served[name] += 1
            family_served[family] += 1
            category_selected[family] += 1
            total += 1
        return selected

    def consume(self, names: list[str]):
        counts = Counter(self.families[name] for name in names)
        self.last_batch_counts = dict(counts)
        if names:
            self.last_batch_share_error = 0.5 * sum(
                abs(counts[family] / len(names) - weight) for family, weight in self.family_weights.items()
            )
        for name in names:
            self.served[name] += 1
            self.family_served[self.families[name]] += 1
            self.total_served += 1

    def _balance_config(self):
        return {
            "targets": self.category_targets,
            "categories": self.families,
            "source_weights": self.weights,
            "tolerance": self.batch_tolerance,
            "surplus": self.max_surplus,
            "batch_size": self.batch_size,
            "algorithm": "mimo_eq6_eq7_hierarchical_v1",
            "target_mix": self.target_mix,
            "oversampling": [self.min_oversampling, self.oversampling, self.max_oversampling],
            "acceptance_floor": self.acceptance_floor,
            "inventory_factor": self.inventory_factor,
        }

    def state_dict(self):
        return {
            "balance_config": self._balance_config(),
            "last_batch_counts": self.last_batch_counts,
            "round_robin_credit": self.round_robin_credit.copy(),
            "scope_round_robin_credit": self.scope_round_robin_credit.copy(),
            "startup_dispatched": self.startup_dispatched,
            "served": dict(self.served),
            "duration": self.duration.copy(),
            "yield_rate": self.yield_rate.copy(),
            "last_batch_share_error": self.last_batch_share_error,
        }

    def load_state_dict(self, state):
        if (
            state.get("balance_config", self._balance_config() if not self.category_targets else None)
            != self._balance_config()
        ):
            raise ValueError("Scheduler balance configuration changed on resume")
        self.last_batch_counts = dict(state.get("last_batch_counts", {}))
        self.round_robin_credit = dict(state.get("round_robin_credit", dict.fromkeys(self.sources, 0.0)))
        if set(self.round_robin_credit) != set(self.sources):
            raise ValueError("Scheduler credit sources changed on resume")
        self.scope_round_robin_credit = dict(
            state.get("scope_round_robin_credit", dict.fromkeys(set(self.scopes.values()), 0.0))
        )
        if set(self.scope_round_robin_credit) != set(self.scopes.values()):
            raise ValueError("Scheduler scope credits changed on resume")
        self.startup_dispatched = state.get("startup_dispatched", 0)
        if set(state["duration"]) != set(self.sources) or set(state["yield_rate"]) != set(self.sources):
            raise ValueError("Scheduler sources changed on resume")
        if set(state["served"]) - set(self.sources):
            raise ValueError("Unknown trained source in scheduler checkpoint")
        self.served = Counter(state["served"])
        self.total_served = sum(self.served.values())
        self.family_served = Counter()
        for name, count in self.served.items():
            self.family_served[self.families[name]] += count
        self.last_batch_share_error = state.get("last_batch_share_error", 0.0)
        self.duration = dict(state["duration"])
        self.yield_rate = dict(state["yield_rate"])

    def metrics(self):
        result = {
            f"scheduler/blocked/{reason}": float(self.last_block == reason)
            for reason in (
                "ready_inventory",
                "buffered_tokens",
                "unfinished_inventory",
                "family_ready_inventory",
                "source_group_budget",
                "no_generation_demand",
            )
        }
        result["scheduler/mean_oversampling"] = self.last_mean_oversampling
        result["scheduler/batch_balance_blocked"] = float(self.batch_balance_blocked)
        result["scheduler/batch_share_total_variation"] = self.last_batch_share_error
        for family, weight in self.family_weights.items():
            result[f"scheduler/batch_shortfall/{family}"] = self.batch_shortfall.get(family, 0)
            result[f"scheduler/batch_share/{family}"] = self.last_batch_counts.get(family, 0) / max(
                1, sum(self.last_batch_counts.values())
            )
            result[f"scheduler/trained_share/{family}"] = self.family_served[family] / max(1, self.total_served)
            result[f"scheduler/target_share/{family}"] = weight
            result[f"scheduler/debt/{family}"] = self.total_served * weight - self.family_served[family]
        for scope in self.scope_round_robin_credit:
            result[f"scheduler/group_budget/{scope}"] = self.last_group_budgets.get(scope, 0)
            result[f"scheduler/launch_share/{scope}"] = self.last_scheduling_weights.get(scope, 0)
        for name in self.sources:
            result[f"scheduler/duration_seconds/{name}"] = self.duration[name]
            result[f"scheduler/usable_yield/{name}"] = self.yield_rate[name]
        return result
