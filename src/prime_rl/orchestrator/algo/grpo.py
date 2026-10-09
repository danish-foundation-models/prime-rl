from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm, iter_trainable_traces
from prime_rl.orchestrator.algo.behavior import detect_behavior
from prime_rl.orchestrator.algo.routing import assign_advantages

if TYPE_CHECKING:
    from prime_rl.orchestrator.clients import InferenceClient


class GRPOAlgorithm(Algorithm):
    """Group Relative Policy Optimization: sample a group of rollouts from the
    policy per example; credit = reward minus the group mean (optionally
    length-shaped); action tokens feed the ``rl`` loss."""

    def __init__(self, config: GRPOAlgoConfig, clients: InferenceClient):
        super().__init__(config, clients)
        self.length_penalty = config.length_penalty
        self.behavior_penalty = config.behavior_penalty

    async def score_group(self, episodes: list[vf.Episode]) -> None:
        traces = [trace for _, trace in iter_trainable_traces(episodes)]
        rewards = torch.tensor([trace.reward for trace in traces], dtype=torch.float32)
        length_penalty = self.length_penalty
        shaped_rewards = rewards.clone()
        if length_penalty is not None:
            output = torch.tensor([trace.num_output_tokens for trace in traces], dtype=rewards.dtype)
            total = torch.tensor([trace.num_total_tokens for trace in traces], dtype=rewards.dtype)
            turns = torch.tensor([trace.num_turns for trace in traces], dtype=rewards.dtype)
            input = total - output
            penalty_frac = (
                length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + length_penalty.num_input_tokens_weight * (input / input.max().clamp(min=1))
                + length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            penalty = rewards.mean() * penalty_frac
            shaped_rewards = rewards - penalty
        if self.behavior_penalty is not None:
            for i, trace in enumerate(traces):
                flags = detect_behavior(trace)
                deduction = self.behavior_penalty.penalty if flags.penalize else 0.0
                shaped_rewards[i] = (shaped_rewards[i] - deduction).clamp(min=0)
                trace.info["behavior_shaping"] = {
                    "malformed_tool_nodes": list(flags.malformed_tool_nodes),
                    "reasoning_only_end": flags.reasoning_only_end,
                    "requested_penalty": deduction,
                    "task_reward": trace.reward,
                    "training_reward": float(shaped_rewards[i]),
                }
                trace.metrics.update(
                    {
                        "behavior/malformed_tool": float(bool(flags.malformed_tool_nodes)),
                        "behavior/reasoning_only_end": float(flags.reasoning_only_end),
                        "behavior/task_reward": trace.reward,
                        "behavior/training_reward": float(shaped_rewards[i]),
                    }
                )
        advantages = shaped_rewards - shaped_rewards.mean()
        for trace, advantage in zip(traces, advantages.tolist(), strict=True):
            assign_advantages(trace, advantage)
