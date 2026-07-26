"""AITER legacy all-reduce for validated gfx90a TP8 BF16 messages."""

import torch

_INSTALL_MARKER = "_prime_rl_gfx90a_legacy_allreduce_v1"
_MIN_MESSAGE_BYTES = 8 * 1024
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def _should_use_legacy_allreduce(
    custom: object,
    world_size: int,
    inp: torch.Tensor,
) -> bool:
    message_bytes = inp.numel() * inp.element_size()
    return (
        custom is not None
        and world_size == 8
        and inp.dtype == torch.bfloat16
        and inp.is_contiguous()
        and _MIN_MESSAGE_BYTES <= message_bytes <= _MAX_MESSAGE_BYTES
        and custom.should_custom_ar(inp)
    )


def install_gfx90a_aiter_legacy_allreduce() -> None:
    """Enable the validated TP8 BF16 legacy all-reduce size window."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.logger import init_logger

    logger = init_logger(__name__)

    if getattr(CudaCommunicator, _INSTALL_MARKER, False):
        return

    original_init = CudaCommunicator.__init__
    original_all_reduce = CudaCommunicator.all_reduce

    def _init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        if "tp" in self.unique_name and self.world_size == 8:
            rocm_aiter_ops.initialize_aiter_allreduce(
                self.cpu_group,
                self.device,
            )
            if rocm_aiter_ops.get_aiter_allreduce() is None:
                raise RuntimeError("AITER legacy all-reduce failed to initialize")

    def _all_reduce(
        self: CudaCommunicator,
        inp: torch.Tensor,
    ) -> torch.Tensor:
        custom = rocm_aiter_ops.get_aiter_allreduce()
        if _should_use_legacy_allreduce(custom, self.world_size, inp):
            if not getattr(self, "_prime_logged_legacy_allreduce", False):
                logger.warning(
                    "Using legacy AITER all-reduce for TP8 BF16 messages "
                    "from %d through %d bytes.",
                    _MIN_MESSAGE_BYTES,
                    _MAX_MESSAGE_BYTES,
                )
                self._prime_logged_legacy_allreduce = True
            output = custom.custom_all_reduce(inp, use_new=False)
            assert output is not None
            return output
        return original_all_reduce(self, inp)

    CudaCommunicator.__init__ = _init
    CudaCommunicator.all_reduce = _all_reduce
    setattr(CudaCommunicator, _INSTALL_MARKER, True)
