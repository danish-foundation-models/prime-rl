"""Reference-gradient and transport tests for optional score centering."""

import msgspec
import numpy as np
import pytest
import torch

from prime_rl.configs.trainer import ScoreCenteringLossConfig
from prime_rl.trainer.batch import _materialize_bin, _MicroBatchBin, pad_micro_batch, prepare_sample
from prime_rl.trainer.models.layers.lm_head import FusedOutputLinear
from prime_rl.trainer.rl.loss import LossInputs, ScoreCenteringLoss, shift_tensor_left, shift_tensor_right
from prime_rl.transports.batch.types import EncodedTensor, TrainingSample


def _loss(logits, q_log, ids, labels, mode):
    logp = logits.log_softmax(-1)
    inputs = LossInputs(
        trainer_logprobs=logp.gather(-1, labels[:, None]).squeeze(-1),
        inference_logprobs=q_log.gather(-1, labels[:, None]).squeeze(-1),
        ref_logprobs=None,
        advantages=torch.tensor([1.0, -0.7, 0.3]),
        loss_mask=torch.ones(3, dtype=torch.bool),
        trainer_topk_logprobs=logp.gather(-1, ids),
        sampler_topk_logprobs=q_log.gather(-1, ids),
        sampler_topk_ids=ids,
    )
    return ScoreCenteringLoss(ScoreCenteringLossConfig(top_k=ids.shape[-1], importance_sampling=mode)).loss(inputs).loss


@pytest.mark.parametrize("mode", ["none", "tis"])
@pytest.mark.parametrize("k", [3, 9])
def test_sc_gradient_matches_full_tail_distribution(mode, k):
    torch.manual_seed(42)
    logits = torch.randn(3, 9, requires_grad=True)
    q_log = (torch.randn(3, 9) * 3).log_softmax(-1)
    ids = q_log.topk(k).indices
    labels = q_log.argmin(-1)  # sampled token deliberately outside top-k
    actual = torch.autograd.grad(_loss(logits, q_log, ids, labels, mode), logits)[0]
    ref = logits.detach().clone().requires_grad_()
    logp = ref.log_softmax(-1)
    p = logp.detach().exp()
    q = q_log.exp()
    head = torch.zeros_like(p, dtype=torch.bool).scatter(-1, ids, True)
    rho = ((q * ~head).sum(-1) / (p * ~head).sum(-1).clamp_min(1e-6)).unsqueeze(-1)
    qhat = torch.where(head, q, rho * p)

    def w(r):
        return r.clamp_max(2) if mode == "tis" else torch.ones_like(r)

    weighted = qhat * w(p / qhat.clamp_min(1e-30))
    correction = (weighted.detach() * logp).sum(-1)
    token_w = w((logp.detach() - q_log).exp()).gather(-1, labels[:, None]).squeeze(-1)
    token_logp = logp.gather(-1, labels[:, None]).squeeze(-1)
    expected_loss = -(torch.tensor([1.0, -0.7, 0.3]) * (token_w * token_logp - correction)).sum()
    expected = torch.autograd.grad(expected_loss, ref)[0]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("mode", ["none", "tis"])
def test_constant_reward_has_zero_expected_gradient(mode):
    torch.manual_seed(3)
    logits = torch.randn(3, requires_grad=True)
    q = torch.tensor([0.8, 0.15, 0.05])
    lp = logits.log_softmax(-1).expand(3, -1)
    inputs = LossInputs(
        lp.diag(),
        q.log(),
        None,
        torch.ones(3),
        torch.ones(3, dtype=torch.bool),
        loss_weights=q,
        trainer_topk_logprobs=lp,
        sampler_topk_logprobs=q.log().expand(3, -1),
        sampler_topk_ids=torch.arange(3).expand(3, -1),
    )
    loss = ScoreCenteringLoss(ScoreCenteringLossConfig(top_k=3, importance_sampling=mode)).loss(inputs).loss
    torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.zeros(3), atol=2e-6, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("selected", [False, True])
def test_chunked_head_matches_dense_gradient_across_vocab_chunks(selected, device):
    torch.manual_seed(5)
    with torch.device(device):
        head = FusedOutputLinear(4, 8201, chunk_size=2).float()
        h = torch.randn(1, 3, 4, requires_grad=True)
        labels = torch.tensor([[8199, 1, 8192]])
        ids = torch.tensor([[[0, 8192, 8200], [8199, 7, 2], [-1, -1, -1]]]) if selected else None
        temperature = torch.tensor([[0.7, 1.0, 1.4]])
        out = head(h, labels, temperature, selected_token_ids=ids)
        objective = (out["logprobs"] * torch.tensor([[1.0, -0.3, 0.5]])).sum()
        weights = torch.randn(1, 3, 3)
        if selected:
            objective = objective + (out["selected_logprobs"] * weights).sum()
        gh, gw = torch.autograd.grad(objective, (h, head.weight))
        rh = h.detach().clone().requires_grad_()
        rw = head.weight.detach().clone().requires_grad_()
        lp = ((rh @ rw.t()) / temperature.unsqueeze(-1)).log_softmax(-1)
        target = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        baseline = (target * torch.tensor([[1.0, -0.3, 0.5]])).sum()
        if selected:
            expected_head = torch.where(ids >= 0, lp.gather(-1, ids.clamp_min(0)), 0)
            torch.testing.assert_close(out["selected_logprobs"], expected_head, atol=2e-6, rtol=2e-5)
            baseline = baseline + (expected_head * weights).sum()
        bgh, bgw = torch.autograd.grad(baseline, (rh, rw))
        torch.testing.assert_close(gh, bgh, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(gw, bgw, atol=2e-6, rtol=2e-5)


def _encoded(a):
    return EncodedTensor(dtype=str(a.dtype), shape=list(a.shape), data=a.tobytes())


def test_topk_transport_truncate_pack_pad_and_label_shift():
    ids = np.arange(10, dtype=np.int32).reshape(5, 2)
    logps = -np.arange(10, dtype=np.float32).reshape(5, 2)
    sample = TrainingSample(
        token_ids=[1, 2, 3, 4, 5],
        mask=[False, True, True, False, True],
        logprobs=[0] * 5,
        temperatures=[1] * 5,
        env_name="test",
        advantages=[1] * 5,
        sampler_topk_ids=_encoded(ids),
        sampler_topk_logprobs=_encoded(logps),
    )
    sample = msgspec.msgpack.decode(msgspec.msgpack.encode(sample), type=TrainingSample)
    one = prepare_sample(sample, 3)
    other = prepare_sample(
        TrainingSample(token_ids=[7, 8], mask=[False, False], logprobs=[0, 0], temperatures=[1, 1], env_name="test"), 3
    )
    packed = _materialize_bin(_MicroBatchBin(samples=[one, other], length=5))
    packed = pad_micro_batch(packed, 8)
    result = np.frombuffer(packed.sampler_topk_ids.data, dtype=np.int32).reshape(8, 2)
    np.testing.assert_array_equal(result[:3], ids[:3])
    assert (result[3:] == -1).all()
    tensor = torch.tensor(result).unsqueeze(0)
    shifted = shift_tensor_right(shift_tensor_left(tensor, -1), -1)
    torch.testing.assert_close(shifted[:, 1:], tensor[:, 1:])


def test_sc_missing_evidence_fails_and_masked_dummy_is_safe():
    lp = torch.zeros(2, requires_grad=True)
    inputs = LossInputs(lp, lp, None, torch.ones(2), torch.ones(2, dtype=torch.bool))
    loss = ScoreCenteringLoss(ScoreCenteringLossConfig())
    with pytest.raises(ValueError, match="generation-time"):
        loss.loss(inputs)
    inputs.loss_mask[:] = False
    loss.loss(inputs).loss.backward()
    assert torch.equal(lp.grad, torch.zeros(2))


def test_renderer_head_parser_and_trace_wire_alignment():
    from renderers.client import MalformedGenerateResponseError, _parse_sampler_topk
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.trace import Branch
    from verifiers.v1.types import AssistantMessage, UserMessage

    from prime_rl.orchestrator.trajectories import _encode_sampler_topk

    choice = {
        "logprobs": {
            "content": [
                {"top_logprobs": [{"token": "token_id:8", "logprob": -1.1}, {"token": "token_id:9", "logprob": -1.4}]},
                {
                    "top_logprobs": [
                        {"token": "token_id:10", "logprob": -0.8},
                        {"token": "token_id:11", "logprob": -2.0},
                    ]
                },
            ]
        }
    }
    ids, logps = _parse_sampler_topk(choice, [12, 13], 2)
    assert ids.dtype == np.int32 and logps.dtype == np.float32
    assert np.exp(logps).sum(-1).max() < 1  # head is not renormalized
    node = MessageNode(
        message=AssistantMessage(content="sample"),
        sampled=True,
        token_ids=[3, 12, 13],
        mask=[False, True, True],
        sampler_topk_ids=ids,
        sampler_topk_logprobs=logps,
    )
    node = MessageNode.model_validate(msgspec.msgpack.decode(msgspec.msgpack.encode(node.model_dump())))
    prompt = MessageNode(message=UserMessage(content="prompt"), token_ids=[1, 2], mask=[False, False])
    second = MessageNode(
        message=AssistantMessage(content="next"),
        sampled=True,
        token_ids=[4, 12, 13],
        mask=[False, True, True],
        sampler_topk_ids=ids,
        sampler_topk_logprobs=logps,
    )
    a, b = _encode_sampler_topk(Branch(index=0, nodes=[prompt, node, prompt, second]))
    out = np.frombuffer(a.data, dtype=np.int32).reshape(a.shape)
    np.testing.assert_array_equal(out[[3, 4, 8, 9]], np.concatenate([ids, ids]))
    assert (out[[0, 1, 2, 5, 6, 7]] == -1).all()
    with pytest.raises(MalformedGenerateResponseError):
        _parse_sampler_topk(choice, [12, 13], 3)
    assert _parse_sampler_topk(choice, [12, 13], 0) == (None, None)


def test_sc_combined_config_enables_capture_and_preserves_default():
    from prime_rl.configs.rl import RLConfig

    raw = {
        "model": {"name": "PrimeIntellect/test-model"},
        "trainer": {},
        "inference": {},
        "orchestrator": {
            "renderer": {"name": "default"},
            "train": {"sampling": {}, "source": [{"name": "test", "env": {"taskset": {"id": "reverse-text"}}}]},
        },
    }
    baseline = RLConfig.model_validate(raw)
    assert baseline.trainer.loss.type == "ipo"
    assert not any("score_centering_top_k" in s.sampling.extra_body for s in baseline.orchestrator.train.source)
    raw["trainer"]["loss"] = {"type": "score_centering", "top_k": 32}
    config = RLConfig.model_validate(raw)
    assert config.trainer.loss.importance_sampling == "none"
    assert config.inference.vllm.max_logprobs >= 33
    assert config.inference.vllm.logprobs_mode == "processed_logprobs"
    assert all(s.sampling.extra_body["score_centering_top_k"] == 32 for s in config.orchestrator.train.source)
    raw["orchestrator"]["train"]["sampling"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="untruncated"):
        RLConfig.model_validate(raw)


def test_renderer_extracts_true_head_when_sampled_token_is_first():
    from renderers.client import _parse_sampler_topk

    choice = {
        "logprobs": {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "token_id:99", "logprob": -9.0},
                        {"token": "token_id:1", "logprob": -1.0},
                        {"token": "token_id:2", "logprob": -2.0},
                    ]
                }
            ]
        }
    }
    ids, logps = _parse_sampler_topk(choice, [99], 2)
    np.testing.assert_array_equal(ids, [[1, 2]])
    np.testing.assert_array_equal(logps, [[-1.0, -2.0]])
