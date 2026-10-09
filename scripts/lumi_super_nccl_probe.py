"""Exercise the native weight wire protocol across LUMI nodes without model allocation."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.inference.vllm.worker.nccl import receive_integer, receive_state_dict
from prime_rl.transports.weights.nccl import broadcast_integer, broadcast_state_dict
from prime_rl.utils.weight_broadcast import WeightBroadcastCommunicator


class BroadcastWarmupCommunicator(PyNcclCommunicator):
    """Probe whether a broadcast-only communicator avoids the all-reduce stall."""

    def __init__(self, *args, **kwargs):
        self.warming_up = True
        super().__init__(*args, **kwargs)
        self.warming_up = False

    def all_reduce(self, tensor, *args, **kwargs):
        if self.warming_up:
            self.broadcast(tensor, src=0)
            return tensor
        return super().all_reduce(tensor, *args, **kwargs)


def worker(local_rank, host, port, output, sender):
    torch.cuda.set_device(local_rank)
    if (
        os.environ.get("PRIME_RL_WEIGHT_HIGH_PRIORITY") == "1"
        and os.environ.get("PRIME_RL_PROBE_NATIVE_WRAPPER") != "1"
    ):
        # vLLM tracks torch.cuda.set_stream; setting it before construction also
        # puts the communicator's warm-up collective on this stream.
        stream = torch.cuda.Stream(device=local_rank, priority=-1)
        torch.cuda.set_stream(stream)
    print(
        json.dumps(
            {
                "local_rank": local_rank,
                "stream_priority": torch.cuda.current_stream().priority,
                "dmabuf": os.environ.get("NCCL_DMABUF_ENABLE"),
                "hw_queues": os.environ.get("GPU_MAX_HW_QUEUES"),
            }
        ),
        flush=True,
    )
    rank = 0 if sender else local_rank + 1
    group = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=9, store_timeout=180)
    comm_class = (
        BroadcastWarmupCommunicator if os.environ.get("PRIME_RL_PROBE_BROADCAST_WARMUP") == "1" else PyNcclCommunicator
    )
    if os.environ.get("PRIME_RL_PROBE_NATIVE_WRAPPER") == "1":
        comm_class = WeightBroadcastCommunicator
    env_before = {k: os.environ.get(k) for k in ("NCCL_ALGO", "NCCL_PROTO")}
    comm = comm_class(group, device=local_rank)
    assert env_before == {k: os.environ.get(k) for k in env_before}, "Weight communicator leaked environment"
    print(
        "WIRE_COMM_READY " + json.dumps({"rank": rank, "caller_priority": torch.cuda.current_stream().priority}),
        flush=True,
    )
    measurements = []
    retained = []
    for iteration in range(5):
        torch.cuda.synchronize()
        started = time.monotonic()
        if sender:
            tensors = {
                "probe.bf16": torch.full((4096, 8192), iteration + 1, dtype=torch.bfloat16, device="cuda"),
                "probe.bf16_small": torch.full((16,), iteration + 1, dtype=torch.bfloat16, device="cuda"),
                "probe.fp32": torch.full((1024,), iteration + 0.25, dtype=torch.float32, device="cuda"),
            }
            broadcast_integer(1, comm)
            broadcast_state_dict(tensors, comm)
        else:
            assert receive_integer(comm) == 1
            found = []
            for name, tensor in receive_state_dict(comm):
                expected = iteration + (1 if name.startswith("probe.bf16") else 0.25)
                assert bool(torch.all(tensor == expected)), f"Corrupt weight payload: {name} rank={rank}"
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(), (
                    f"Received weight retains unrelated bucket storage: {name}"
                )
                if name == "probe.bf16_small":
                    retained.append(tensor)
                found.append(name)
            assert found == ["probe.bf16", "probe.bf16_small", "probe.fp32"], found
            assert sum(t.untyped_storage().nbytes() for t in retained) == 32 * (iteration + 1)
            assert all(bool(torch.all(t == index + 1)) for index, t in enumerate(retained))
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        measurements.append({"iteration": iteration, "seconds": seconds, "payload_bytes": 67112992})
        print("WIRE_PROBE " + json.dumps({"rank": rank, **measurements[-1]}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank-{rank}.json").write_text(
        json.dumps({"passed": True, "rank": rank, "measurements": measurements}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=29611)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sender", action="store_true")
    args = parser.parse_args()
    if args.sender:
        worker(0, args.host, args.port, args.output, True)
    else:
        mp.spawn(worker, args=(args.host, args.port, args.output, False), nprocs=8, join=True)
