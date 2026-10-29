"""Detect degenerate sampled turns: long runs of near-random tokens.

At temperature 1 with no tail cut-off, a rare sampled token can derail a long agent
context into thousands of near-random tokens. The sampler's own logprobs show it: a
normal turn averages well under 1 nat per token, a derailed one several.

The per-turn rule lives in ``verifiers.v1.degenerate`` so the interception server's live
stop (``RolloutLimits.degenerate_turn``, stop condition ``degenerate_turn``) and this
post-hoc check reach the same verdict on the same turn. The post-hoc check stays the
authority: the dispatcher drops whatever it flags. Turns the engine ended by repetition
detection (finish reason ``repetition``) are exempt in both.
"""

from verifiers.v1.degenerate import DEGENERATE_TURN, DegenerateTurn, find_degenerate_turn

__all__ = ["DEGENERATE_TURN", "DegenerateTurn", "find_degenerate_turn"]
