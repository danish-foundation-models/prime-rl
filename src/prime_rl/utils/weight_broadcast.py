"""Run-scoped RCCL settings for the separate trainer-to-inference communicator."""

import os
from contextlib import contextmanager

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


@contextmanager
def _weight_transport_environment():
    overrides = {
        name: os.environ[f"PRIME_RL_WEIGHT_{name}"]
        for name in ("NCCL_ALGO", "NCCL_PROTO")
        if f"PRIME_RL_WEIGHT_{name}" in os.environ
    }
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class WeightBroadcastCommunicator(PyNcclCommunicator):
    """Keep weight collectives on their own high-priority stream when requested."""

    def __init__(self, group, device):
        self._initializing_weight_communicator = True
        self.weight_stream = None
        with _weight_transport_environment():
            if isinstance(group, StatelessProcessGroup):
                channel_settings = {
                    name: os.environ.get(name)
                    for name in ("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS")
                }
                peer_settings = group.all_gather_obj(channel_settings)
                if any(settings != peer_settings[0] for settings in peer_settings):
                    raise RuntimeError(
                        "Weight broadcast requires identical RCCL channel settings from process startup: "
                        f"{dict(enumerate(peer_settings))}"
                    )
            if os.environ.get("PRIME_RL_WEIGHT_HIGH_PRIORITY") == "1":
                self.weight_stream = torch.cuda.Stream(device=device, priority=-1)
                with torch.cuda.stream(self.weight_stream):
                    super().__init__(group, device=device)
            else:
                super().__init__(group, device=device)

        self._initializing_weight_communicator = False

    def all_reduce(self, in_tensor, out_tensor=None, op=None, stream=None):
        if self._initializing_weight_communicator:
            # Warm the operation this communicator uses, above RCCL's tiny-message range.
            warmup = torch.zeros(256, dtype=torch.float32, device=self.device)
            self.broadcast(warmup, src=0, stream=stream)
            return in_tensor
        if op is None:
            return super().all_reduce(in_tensor, out_tensor=out_tensor, stream=stream)
        return super().all_reduce(in_tensor, out_tensor=out_tensor, op=op, stream=stream)

    def broadcast(self, tensor, src, stream=None):
        if tensor.numel() * tensor.element_size() < 1024:
            caller = stream if stream is not None else torch.cuda.current_stream(self.device)
            with torch.cuda.stream(caller):
                padded = torch.zeros(1024 // tensor.element_size(), dtype=tensor.dtype, device=tensor.device)
                if self.rank == src:
                    padded[:tensor.numel()].copy_(tensor.reshape(-1))
                self._broadcast_payload(padded, src, stream=caller)
                tensor.copy_(padded[:tensor.numel()].view_as(tensor))
            return
        return self._broadcast_payload(tensor, src, stream=stream)

    def _broadcast_payload(self, tensor, src, stream=None):
        if self.weight_stream is None:
            return super().broadcast(tensor, src, stream=stream)
        caller = stream if stream is not None else torch.cuda.current_stream(self.device)
        self.weight_stream.wait_stream(caller)
        super().broadcast(tensor, src, stream=self.weight_stream)
        tensor.record_stream(self.weight_stream)
        caller.wait_stream(self.weight_stream)
