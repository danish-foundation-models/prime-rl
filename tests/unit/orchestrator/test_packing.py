import asyncio
import random
from types import SimpleNamespace

import msgspec
import numpy as np

from prime_rl.orchestrator.packing import BatchPacker
from prime_rl.transports.batch.types import EncodedTensor, MicroBatch, TrainingSample


def _sample(length: int, rng: random.Random) -> TrainingSample:
    topk_ids = np.arange(length * 4, dtype=np.int32).reshape(length, 4)
    topk_logprobs = np.full((length, 4), -1.5, dtype=np.float32)
    return TrainingSample(
        token_ids=[rng.randrange(1000) for _ in range(length)],
        mask=[i % 3 == 0 for i in range(length)],
        logprobs=[-0.1] * length,
        temperatures=[1.0] * length,
        advantages=[0.5] * length,
        env_name="test-env",
        trace_id=f"t{length}",
        branch_index=0,
        sampler_topk_ids=EncodedTensor(dtype="int32", shape=[length, 4], data=topk_ids.tobytes()),
        sampler_topk_logprobs=EncodedTensor(dtype="float32", shape=[length, 4], data=topk_logprobs.tobytes()),
    )


def test_process_packing_matches_in_process_packing():
    config = SimpleNamespace(
        seq_len=512,
        num_train_workers=2,
        pad_to_multiple_of=8,
        model=SimpleNamespace(name="/nonexistent/model"),  # falls back to token-count bin costs
        tokenizer=SimpleNamespace(trust_remote_code=False),
    )
    packer = BatchPacker(config)
    rng = random.Random(0)
    samples = [_sample(rng.randint(20, 400), rng) for _ in range(12)]
    try:
        buffers = asyncio.run(packer.pack_encoded(samples))
    finally:
        packer.close()
    expected = packer.pack(samples)
    encoder = msgspec.msgpack.Encoder()
    assert buffers == [encoder.encode(micro_batches) for micro_batches in expected]
    decoded = msgspec.msgpack.Decoder(type=list[MicroBatch]).decode(buffers[0])
    assert decoded and all(len(mb.input_ids) % 8 == 0 for mb in decoded)
