"""Compare frozen rollout logprobs with CP/EP trainer forwards, with and without replay."""

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import prime_rl._compat  # noqa: F401
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.model import forward, setup_model
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.rl.loss import shift_tensor_left
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.cp import gather_for_cp_wo_grad, setup_context_parallel, setup_cp_params, shard_for_cp
from prime_rl.utils.logger import setup_logger


def main(args):
    config = TrainerConfig.model_validate_json(args.config.read_text())
    setup_logger("info")
    setup_torch_distributed(timeout=timedelta(seconds=1800))
    torch.set_float32_matmul_precision(config.matmul_precision)
    resolve_ep(config.model)
    dims = get_parallel_dims(config.model)
    model = setup_model(config.model, dims, False)
    setup_context_parallel(model, config.model, dims)
    cp_rank = dims.world_mesh["cp"].get_local_rank()
    cp_group = dims.world_mesh["cp"].get_group()
    report = {"cases": [], "optimizer_updates": 0, "cp": dims.cp, "ep": config.model.ep}
    for name in ("short", "multiturn", "long"):
        sample = np.load(args.samples / f"{name}-0.npz")
        tokens = sample["token_ids"]
        logprobs = sample["completion_logprobs"]
        prompt_len = len(tokens) - len(logprobs)
        length = ((len(tokens) + 511) // 512) * 512
        ids = torch.zeros((1, length), dtype=torch.long, device="cuda")
        ids[0, : len(tokens)] = torch.as_tensor(tokens, device="cuda")
        positions = torch.arange(length, device="cuda").unsqueeze(0)
        seq_lens = torch.tensor([length], dtype=torch.int32, device="cuda")
        labels = shard_for_cp(shift_tensor_left(ids), cp_rank=cp_rank, cp_world_size=dims.cp)
        captured = torch.as_tensor(sample["routes"].astype(np.int64), device="cuda")
        # The last sampled token has no next-token forward; padding is loss-masked.
        routes = captured[-1:].expand(length, -1, -1).clone()
        routes[: len(captured)] = captured
        routes = shard_for_cp(routes.unsqueeze(0), cp_rank=cp_rank, cp_world_size=dims.cp)
        ids, positions = setup_cp_params(
            ids, positions, cp_rank, dims.cp, cp_group, seq_lens=seq_lens, cp_style=config.model.cp_style
        )
        for replay in (False, True):
            dist.barrier()
            started = time.monotonic()
            with torch.no_grad():
                out = forward(
                    model,
                    ids,
                    positions,
                    seq_lens=seq_lens,
                    labels=labels,
                    temperature=torch.ones_like(labels, dtype=torch.float32),
                    routed_experts=routes if replay else None,
                    seq_lens_are_pre_shard=True,
                )
                full = gather_for_cp_wo_grad(out["logprobs"], dims.cp, cp_group)
                actual = full[0, prompt_len - 1 : len(tokens) - 1].float().cpu().numpy()
            delta = actual.astype(np.float64) - logprobs
            assert np.isfinite(delta).all(), f"Nonfinite logprobs: {name=} {replay=}"
            case = {
                "name": name,
                "replay": replay,
                "seconds": time.monotonic() - started,
                "tokens": len(tokens),
                "mean_abs_logprob_delta": float(np.abs(delta).mean()),
                "p95_abs_logprob_delta": float(np.quantile(np.abs(delta), 0.95)),
                "max_abs_logprob_delta": float(np.abs(delta).max()),
                "mean_mismatch_kl": float((np.expm1(delta) - delta).mean()),
            }
            report["cases"].append(case)
            if get_world().is_master:
                args.output.mkdir(parents=True, exist_ok=True)
                np.savez(args.output / f"{name}-replay{int(replay)}.npz", trainer=actual, inference=logprobs)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print("PARITY_CASE " + json.dumps(case), flush=True)
            del out, full
        if name == "short":
            # Exercise the replay backward path without updating or exporting weights.
            mask = torch.zeros((1, length), dtype=torch.bool, device="cuda")
            mask[:, prompt_len - 1 : len(tokens) - 1] = True
            local_mask = shard_for_cp(mask, cp_rank=cp_rank, cp_world_size=dims.cp)
            out = forward(
                model,
                ids,
                positions,
                seq_lens=seq_lens,
                labels=labels,
                temperature=torch.ones_like(labels, dtype=torch.float32),
                routed_experts=routes,
                seq_lens_are_pre_shard=True,
            )
            loss = -(out["logprobs"] * local_mask).sum() / len(logprobs)
            loss.backward()
            bad = torch.zeros((), dtype=torch.int64, device="cuda")
            grad_params = torch.zeros_like(bad)
            for param in model.parameters():
                if param.grad is None:
                    continue
                grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
                bad += (~torch.isfinite(grad)).any().to(torch.int64)
                grad_params += 1
            dist.all_reduce(bad)
            dist.all_reduce(grad_params)
            assert bad.item() == 0 and grad_params.item() > 0, "Replay backward gradient failure"
            model.zero_grad(set_to_none=True)
            report["replay_backward_finite"] = True
            del out
    report["completed"] = True
    if get_world().is_master:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("REPLAY_PARITY_COMPLETE", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
