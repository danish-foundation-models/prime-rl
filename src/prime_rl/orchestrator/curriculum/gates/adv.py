"""Advantage-based training-sample admission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.gates.base import AdmissionGate
from prime_rl.orchestrator.trajectories import iter_trainable_branches

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import AdvRangeGateConfig


class AdvRangeGate(AdmissionGate):
    """Reject groups whose trainable-token advantages all fall inside a range.

    The default ``[0, 0]`` interval filters groups with no online learning
    signal. Auxiliary CE/reference-KL supervision survives this gate, as do
    groups without an advantage stream.
    """

    def __init__(self, config: AdvRangeGateConfig) -> None:
        self.config = config

    def admit(self, group: list[vf.Episode]) -> bool:
        # This gate only removes groups with no useful RL credit. Do not throw
        # away independent supervision already attached by ECHO or distillation.
        for episode in group:
            for trace in episode.traces:
                if trace.has_error or not trace.agent.trainable:
                    continue
                if any(
                    weight != 0
                    for branch, _ in iter_trainable_branches(trace)
                    for node in branch.nodes
                    for component in ("ce", "ref_kl")
                    for weight in (node.loss_weights or {}).get(component, [])
                ):
                    return True
        advantages = [
            advantage
            for episode in group
            for trace in episode.traces
            if not trace.has_error and trace.agent.trainable
            for node in trace.nodes
            for advantage in node.advantages or []
        ]
        if not advantages:
            return True
        return not all(self.config.reject_min <= advantage <= self.config.reject_max for advantage in advantages)
