"""Detect degenerate sampled turns: long runs of near-random tokens.

At temperature 1 with no tail cut-off, a rare sampled token can derail a long agent
context into thousands of near-random tokens. The sampler's own logprobs show it: a
normal turn averages well under 1 nat per token, a derailed one several.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DegenerateTurn:
    index: int
    tokens: int
    mean_nll: float


def find_degenerate_turn(trace, *, min_tokens: int, max_mean_nll: float) -> DegenerateTurn | None:
    """The first sampled turn of at least ``min_tokens`` whose mean negative logprob exceeds ``max_mean_nll``."""
    index = 0
    for node in getattr(trace, "nodes", None) or []:
        if not getattr(node, "sampled", False):
            continue
        logprobs = [lp for lp in (getattr(node, "logprobs", None) or []) if isinstance(lp, (int, float))]
        if len(logprobs) >= min_tokens:
            mean_nll = -sum(logprobs) / len(logprobs)
            if mean_nll > max_mean_nll:
                return DegenerateTurn(index=index, tokens=len(logprobs), mean_nll=mean_nll)
        index += 1
    return None
