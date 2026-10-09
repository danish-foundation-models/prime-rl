"""Bounded work admission, aggregating harness variants by taskset."""
import math
from collections import defaultdict


def eligible_sources(sources: list[tuple[str, str, float, int]], pressure: dict[str, int], target: int) -> set[str]:
    weights: dict[str, float] = defaultdict(float)
    used: dict[str, int] = defaultdict(int)
    group_sizes: dict[str, int] = defaultdict(int)
    for name, family, weight, group_size in sources:
        weights[family] += weight
        used[family] += pressure.get(name, 0)
        group_sizes[family] = max(group_sizes[family], group_size)
    total = sum(weights.values())
    budgets = {family: max(size, math.ceil(target * weights[family] / total / size) * size)
               for family, size in group_sizes.items()}
    return {name for name, family, _, size in sources if used[family] + size <= budgets[family]}
