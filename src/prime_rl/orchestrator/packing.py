import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import msgspec
from transformers import AutoConfig

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.trainer.batch import build_bin_cost, prepare_batch
from prime_rl.transports.batch.types import MicroBatch, TrainingSample
from prime_rl.utils.logger import get_logger


class BatchPacker:
    """Bin-packs a step's training samples into one micro-batch list per trainer DP rank."""

    def __init__(self, config: OrchestratorConfig):
        self.seq_len = config.seq_len
        self.num_train_workers = config.num_train_workers
        self.pad_to_multiple_of = config.pad_to_multiple_of
        self.model_name = config.model.name
        self.trust_remote_code = config.tokenizer.trust_remote_code
        self.bin_cost = build_bin_cost(load_model_config(self.model_name, self.trust_remote_code))
        self._executor: ProcessPoolExecutor | None = None

    def pack(self, samples: list[TrainingSample]) -> list[list[MicroBatch]]:
        return prepare_batch(
            rollouts=samples,
            seq_len=self.seq_len,
            num_train_workers=self.num_train_workers,
            bin_cost=self.bin_cost,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    async def pack_encoded(self, samples: list[TrainingSample]) -> list[bytes]:
        """Pack and msgpack-encode one buffer per DP rank in a worker process.

        Packing and encoding a large agentic batch is seconds of pure Python and
        multi-GB copies. In a thread it competes with the event loop for the GIL;
        a busy orchestrator stalled for minutes that way and starved inference of
        requests. Samples cross the process boundary as per-sample msgpack
        buffers (encoded one at a time so the loop runs between them) and the
        ranks come back as bytes, ready for the sender.
        """
        encoder = msgspec.msgpack.Encoder()
        buffers = []
        for sample in samples:
            buffers.append(encoder.encode(sample))
            await asyncio.sleep(0)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor_or_create(), _pack_encode, buffers)

    def _executor_or_create(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(
                    self.seq_len,
                    self.num_train_workers,
                    self.pad_to_multiple_of,
                    self.model_name,
                    self.trust_remote_code,
                ),
            )
        return self._executor

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


def load_model_config(model_name: str, trust_remote_code: bool | None):
    try:
        return AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception as e:
        get_logger().warning(
            f"Could not load model config for {model_name} ({e}) - "
            "packing balances by token count instead of estimated FLOPs"
        )
        return None


# Worker-process state, set once by the pool initializer.
_worker: dict = {}


def _init_worker(seq_len, num_train_workers, pad_to_multiple_of, model_name, trust_remote_code) -> None:
    _worker.update(
        seq_len=seq_len,
        num_train_workers=num_train_workers,
        pad_to_multiple_of=pad_to_multiple_of,
        bin_cost=build_bin_cost(load_model_config(model_name, trust_remote_code)),
        decoder=msgspec.msgpack.Decoder(type=TrainingSample),
        encoder=msgspec.msgpack.Encoder(),
    )


def _pack_encode(buffers: list[bytes]) -> list[bytes]:
    samples = [_worker["decoder"].decode(buffer) for buffer in buffers]
    grid = prepare_batch(
        rollouts=samples,
        seq_len=_worker["seq_len"],
        num_train_workers=_worker["num_train_workers"],
        bin_cost=_worker["bin_cost"],
        pad_to_multiple_of=_worker["pad_to_multiple_of"],
    )
    return [_worker["encoder"].encode(micro_batches) for micro_batches in grid]
