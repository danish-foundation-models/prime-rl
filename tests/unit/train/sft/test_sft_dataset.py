import hashlib
import json
from collections import Counter

import pytest
import torch
from datasets import Dataset, Features, Sequence, Value, interleave_datasets
from renderers import create_renderer
from renderers.base import MultiModalData, PlaceholderRange, RenderedTrainingSample
from transformers import AutoTokenizer

import prime_rl.trainer.sft.data as sft_data
from prime_rl.configs.sft import PretokenizedSFTDataConfig
from prime_rl.trainer.sft.data import (
    CatDataset,
    PretokenizedSFTDataset,
    SFTDataset,
    _drop_null_fields,
    load_pretokenized_dataset,
    requires_renderer,
)
from prime_rl.trainer.utils import print_sample
from prime_rl.utils.chat_template import ensure_tokenizer_pad_token

_BOS_TOKEN_ID = 0
_STOP_TOKEN_ID = 1


def test_ensure_tokenizer_pad_token_preserves_distinct_pad_and_falls_back_to_eos():
    distinct = type("Tokenizer", (), {"pad_token_id": 3, "eos_token_id": 68})()
    ensure_tokenizer_pad_token(distinct)
    assert distinct.pad_token_id == 3

    missing = type("Tokenizer", (), {"pad_token_id": None, "eos_token_id": 68})()
    ensure_tokenizer_pad_token(missing)
    assert missing.pad_token_id == 68


@pytest.mark.parametrize(
    ("data_type", "has_validation", "expected"),
    [
        ("sft", False, True),
        ("pretokenized", False, False),
        ("pretokenized", True, True),
        ("fake", False, False),
    ],
)
def test_requires_renderer(data_type, has_validation, expected):
    assert requires_renderer(data_type, has_validation=has_validation) is expected


def test_pretokenized_config_rejects_multiple_dataloader_workers(tmp_path):
    with pytest.raises(ValueError):
        PretokenizedSFTDataConfig(path=tmp_path, num_workers=2)


def _sample_token_ids(value: str) -> list[int]:
    return [ord(char) + 2 for char in value]


@pytest.fixture(scope="module")
def build_dummy_dataset():
    return lambda letter, num_examples: Dataset.from_list(
        [{"messages": [{"role": "assistant", "content": f"{letter}{i}"}]} for i in range(num_examples)]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param('{"reasoning_effort": null}', id="json-string"),
        pytest.param({"reasoning_effort": None}, id="dict"),
    ],
)
def test_drop_null_fields_preserves_tool_call_arguments(arguments):
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Calling a tool", "image_url": None}],
        "tool_calls": [{"function": {"name": "listReasoningModels", "arguments": arguments}}],
        "metadata": {"arguments": {"unrelated_null": None}},
    }

    cleaned = _drop_null_fields(message)

    assert cleaned["tool_calls"][0]["function"]["arguments"] == arguments
    assert cleaned["content"] == [{"type": "text", "text": "Calling a tool"}]
    assert cleaned["metadata"] == {"arguments": {}}


def test_init_sft_dataset(build_dummy_dataset, dummy_renderer):
    """Tests basic initialization."""
    dataset = build_dummy_dataset("a", 1)
    sft_dataset = SFTDataset(dataset, dummy_renderer)
    assert sft_dataset is not None


def test_raise_error_if_no_prompt_and_completion(build_dummy_dataset):
    """Tests that an error is raised if no supported SFT message fields are provided."""
    dataset = Dataset.from_list([{"text": "a0"}])
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    sft_dataset = SFTDataset(dataset, create_renderer(tokenizer))
    with pytest.raises(ValueError):
        next(iter(sft_dataset))


@pytest.mark.parametrize("max_epochs", [1, 2, 4])
def test_sft_first_exhausted(build_dummy_dataset, dummy_renderer, max_epochs: int):
    a = build_dummy_dataset("a", 1)
    b = build_dummy_dataset("b", 2)
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="first_exhausted")
    dataset = SFTDataset(dataset, dummy_renderer, shuffle=False, max_epochs=max_epochs)
    num_samples = 0
    sampling_order = []
    for x in dataset:
        sampling_order.append(x["target_ids"][:-1])
        num_samples += 1
    assert num_samples == max_epochs * min([len(d) for d in ds]) * len(ds)
    assert sampling_order == [_sample_token_ids("a0"), _sample_token_ids("b0")] * max_epochs


@pytest.mark.parametrize("max_epochs", [1, 2, 4])
def test_sft_all_exhausted(build_dummy_dataset, dummy_renderer, max_epochs: int):
    a = build_dummy_dataset("a", 1)
    b = build_dummy_dataset("b", 2)
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="all_exhausted")
    dataset = SFTDataset(dataset, dummy_renderer, shuffle=False, max_epochs=max_epochs)
    num_samples = 0
    sampling_order = []
    for x in dataset:
        sampling_order.append(x["target_ids"][:-1])
        num_samples += 1
    assert num_samples == max_epochs * max([len(d) for d in ds]) * len(ds)
    print(sampling_order)
    assert (
        sampling_order
        == [
            _sample_token_ids("a0"),
            _sample_token_ids("b0"),
            _sample_token_ids("a0"),
            _sample_token_ids("b1"),
        ]
        * max_epochs
    )


@pytest.mark.parametrize(
    "probs",
    [
        pytest.param((0.5, 0.5), id="equal_probs"),
        pytest.param((1 / 10, 9 / 10), id="low_high_probs"),
        pytest.param((9 / 10, 1 / 10), id="high_low_probs"),
    ],
)
def test_sft_all_exhausted_with_probs(build_dummy_dataset, dummy_renderer, probs: list[float]):
    """Tests that the ratio of samples from different datasets is as specified, in expectation."""
    a = build_dummy_dataset("a", int(1e3))
    b = build_dummy_dataset("b", int(10e3))
    ds = [a, b]
    dataset = interleave_datasets(ds, stopping_strategy="all_exhausted", probabilities=probs)
    dataset = SFTDataset(dataset, dummy_renderer, shuffle=False, max_epochs=1)
    num_samples = 0
    sampling_freq = []
    for x in dataset:
        sampling_freq.append(x["target_ids"][0])
        num_samples += 1
    sampling_freq = Counter(sampling_freq)
    ratio_a = sampling_freq[ord("a") + 2] / num_samples
    ratio_b = sampling_freq[ord("b") + 2] / num_samples
    assert ratio_a > probs[0] * 0.8 and ratio_a < probs[0] * 1.2, (
        f"Expected frequency of samples from a to be between {probs[0] * 0.8} and {probs[0] * 1.2}, but got {ratio_a}"
    )
    assert ratio_b > probs[1] * 0.8 and ratio_b < probs[1] * 1.2, (
        f"Exepcted frequency of samples from b to be between {probs[1] * 0.8} and {probs[1] * 1.2}, but got {ratio_b}"
    )


def test_sft_dataset_state(build_dummy_dataset, dummy_renderer):
    """Tests the state of the dataset within and across epochs."""
    dataset = build_dummy_dataset("", 4)
    dataset = SFTDataset(dataset, dummy_renderer, shuffle=False, max_epochs=2)
    dataiter = iter(dataset)

    # Initial state
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Epoch 1
    for i in range(4):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 0, "step": i + 1}

    # Epoch 2
    for i in range(4):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 1, "step": 4 + i + 1}

    with pytest.raises(StopIteration):
        next(dataiter)


def test_sft_dataset_state_resume(build_dummy_dataset, dummy_renderer):
    """Tests resuming the dataset from checkpoint in between epochs."""
    dataset = SFTDataset(
        build_dummy_dataset("", 4),
        dummy_renderer,
        shuffle=False,
        max_epochs=2,
    )
    dataiter = iter(dataset)

    # Initial state
    assert dataset.state_dict() == {"step": 0, "epoch": 0}

    # Epoch 1
    for i in range(4):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 0, "step": i + 1}

    # Resuming from checkpoint cross epoch
    state_dict = dataset.state_dict()
    del dataset
    dataset = SFTDataset(
        build_dummy_dataset("", 4),
        dummy_renderer,
        shuffle=False,
        max_epochs=2,
    )
    dataset.load_state_dict(state_dict)
    dataiter = iter(dataset)

    # Epoch 2.1
    for i in range(2):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 1, "step": 4 + i + 1}

    # Resuming from checkpoint mid epoch
    state_dict = dataset.state_dict()
    del dataset
    dataset = SFTDataset(
        build_dummy_dataset("", 4),
        dummy_renderer,
        shuffle=False,
        max_epochs=2,
    )
    dataset.load_state_dict(state_dict)
    dataiter = iter(dataset)

    # Epoch 2.2
    for i in range(2, 4):
        sample = next(dataiter)
        assert sample["target_ids"][:-1] == _sample_token_ids(str(i))
        assert dataset.state_dict() == {"epoch": 1, "step": 4 + i + 1}

    with pytest.raises(StopIteration):
        next(dataiter)


def test_multiturn_loss_mask():
    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "system", "content": "System 0"}, {"role": "user", "content": "Prompt 0"}],
                "completion": [
                    {"role": "assistant", "content": "Completion 0"},
                    {"role": "user", "content": "Prompt 1"},
                    {"role": "assistant", "content": "Completion 1"},
                ],
            },
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")  # Properly handles multi-turn think
    dataset = SFTDataset(dataset, create_renderer(tokenizer), max_examples=1)
    sample = next(iter(dataset))
    print_sample(sample["input_ids"], sample["loss_mask"], tokenizer)


def test_multiturn_loss_mask_with_tools():
    tool_example = {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What's the weather like in San Francisco and New York?"},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "San Francisco, CA"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "New York, NY"}'},
                    },
                ],
            },
            {"role": "tool", "content": '{"temperature": 65, "condition": "Sunny"}', "tool_call_id": "call_1"},
            {"role": "tool", "content": '{"temperature": 45, "condition": "Cloudy"}', "tool_call_id": "call_2"},
            {
                "role": "assistant",
                "content": "Based on the weather data:\n\n**San Francisco, CA**: It's currently 65°F and sunny - perfect weather!\n\n**New York, NY**: It's 45°F and cloudy - you might want to bring a jacket.",
            },
            {"role": "user", "content": "Should I pack an umbrella for New York?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "get_precipitation_forecast",
                            "arguments": '{"location": "New York, NY", "days": 3}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": '{"forecast": [{"day": 1, "chance_of_rain": 20}, {"day": 2, "chance_of_rain": 60}, {"day": 3, "chance_of_rain": 40}]}',
                "tool_call_id": "call_3",
            },
            {
                "role": "assistant",
                "content": "Looking at the 3-day precipitation forecast for New York:\n- Day 1: 20% chance of rain\n- Day 2: 60% chance of rain\n- Day 3: 40% chance of rain\n\nI'd recommend packing an umbrella, especially for day 2 when there's a 60% chance of rain.",
            },
        ],
    }

    dataset = Dataset.from_list([tool_example])
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")  # Properly handles multi-turn think
    dataset = SFTDataset(dataset, create_renderer(tokenizer), max_examples=1)
    sample = next(iter(dataset))
    print_sample(sample["input_ids"], sample["loss_mask"], tokenizer)


def test_messages_rows_are_equivalent_to_empty_prompt_completion():
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What's the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "San Francisco, CA"}'},
                }
            ],
        },
        {"role": "tool", "content": '{"temperature": 65, "condition": "Sunny"}', "tool_call_id": "call_1"},
        {"role": "assistant", "content": "It is 65F and sunny in San Francisco."},
    ]

    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    messages_dataset = SFTDataset(
        Dataset.from_list([{"messages": messages}]),
        create_renderer(tokenizer),
        max_examples=1,
    )
    split_dataset = SFTDataset(
        Dataset.from_list([{"prompt": [], "completion": messages}]),
        create_renderer(tokenizer),
        max_examples=1,
    )

    assert next(iter(messages_dataset)) == next(iter(split_dataset))


def test_messages_take_precedence_over_prompt_and_completion():
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    row = {
        "messages": [
            {"role": "system", "content": "System from messages"},
            {"role": "user", "content": "Prompt from messages"},
            {"role": "assistant", "content": "Completion from messages"},
        ],
        "prompt": [{"role": "user", "content": "Ignored prompt"}],
        "completion": [{"role": "assistant", "content": "Ignored completion"}],
    }

    messages_dataset = SFTDataset(
        Dataset.from_list([row]),
        create_renderer(tokenizer),
        max_examples=1,
    )
    expected_dataset = SFTDataset(
        Dataset.from_list([{"prompt": [], "completion": row["messages"]}]),
        create_renderer(tokenizer),
        max_examples=1,
    )

    assert next(iter(messages_dataset)) == next(iter(expected_dataset))


def test_null_messages_falls_back_to_prompt_and_completion():
    # Arrow schema union adds `messages: None` to prompt/completion rows when
    # other rows in the file have a `messages` column
    tokenizer = AutoTokenizer.from_pretrained("PrimeIntellect/Qwen3-0.6B")
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "content": "4"}]

    mixed_row_dataset = SFTDataset(
        Dataset.from_list([{"messages": None, "prompt": prompt, "completion": completion}]),
        create_renderer(tokenizer),
        max_examples=1,
    )
    expected_dataset = SFTDataset(
        Dataset.from_list([{"prompt": prompt, "completion": completion}]),
        create_renderer(tokenizer),
        max_examples=1,
    )

    assert next(iter(mixed_row_dataset)) == next(iter(expected_dataset))


def test_vlm_truncation_does_not_append_trainable_eos(monkeypatch, dummy_renderer):
    mm = MultiModalData(
        mm_placeholders={"image": [PlaceholderRange(offset=1, length=1)]},
        mm_items={"image": [{"pixel_values": torch.ones(1, 1), "image_grid_thw": torch.tensor([[1, 1, 1]])}]},
    )

    def fake_build_training_sample(*args, **kwargs):
        return RenderedTrainingSample(
            token_ids=[10, 11, 12, _STOP_TOKEN_ID],
            loss_mask=[False, False, False, True],
            multi_modal_data=mm,
            mm_token_type_ids=[0, 1, 0, 0],
        )

    monkeypatch.setattr(sft_data, "build_training_sample", fake_build_training_sample)
    dataset = SFTDataset(Dataset.from_list([]), dummy_renderer, seq_len=2, multimodal=True)

    assert dataset._process({"messages": [{"role": "assistant", "content": "ignored"}]}) is None


def _sft_sample(
    input_ids: list[int],
    *,
    mm_kwargs: dict[str, torch.Tensor] | None = None,
    mm_token_type_ids: list[int] | None = None,
) -> dict:
    return {
        "input_ids": input_ids,
        "position_ids": list(range(len(input_ids))),
        "loss_mask": [True] * len(input_ids),
        "target_ids": [x + 1 for x in input_ids],
        "seq_lens": [len(input_ids)],
        "mm_kwargs": mm_kwargs,
        "mm_token_type_ids": mm_token_type_ids,
    }


def test_pretokenized_dataset_preserves_packed_boundaries():
    dataset = PretokenizedSFTDataset(
        Dataset.from_list(
            [
                {
                    "input_ids": [10, 11, 20, 21, 0],
                    "target_ids": [11, 12, 21, 22, 0],
                    "loss_mask": [0, 1, 1, 1, 0],
                    "seq_lens": [2, 2, 1],
                }
            ]
        ),
        shuffle=False,
        seq_len=5,
    )

    sample = next(iter(dataset))

    assert sample["input_ids"] == [10, 11, 20, 21, 0]
    assert sample["target_ids"] == [11, 12, 21, 22, 0]
    assert sample["loss_mask"] == [False, True, True, True, False]
    assert sample["seq_lens"] == [2, 2, 1]
    assert sample["position_ids"] == [0, 1, 0, 1, 0]


class _DummyBackendTokenizer:
    def __init__(self, payload: str):
        self.payload = payload

    def to_str(self):
        return self.payload


class _DummyPretokenizedTokenizer:
    bos_token_id = None
    eos_token_id = 68
    pad_token_id = 3
    unk_token_id = None

    def __init__(self, backend_payload: str):
        self.backend_tokenizer = _DummyBackendTokenizer(backend_payload)


def _write_pretokenized_artifact(tmp_path, *, declared_rows: int = 1, loss_mask_dtype: str = "uint8"):
    root = tmp_path / "packed"
    shard_path = root / "chunk_00000"
    backend_payload = '{"model":"test"}'
    features = Features(
        {
            "input_ids": Sequence(Value("uint32"), length=4),
            "target_ids": Sequence(Value("uint32"), length=4),
            "loss_mask": Sequence(Value(loss_mask_dtype), length=4),
            "seq_lens": Sequence(Value("int64")),
        }
    )
    Dataset.from_dict(
        {
            "input_ids": [[10, 11, 20, 0]],
            "target_ids": [[11, 12, 21, 0]],
            "loss_mask": [[1, 1, 1, 0]],
            "seq_lens": [[2, 1, 1]],
        },
        features=features,
    ).save_to_disk(shard_path)
    files = [
        {
            "path": path.relative_to(shard_path).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(shard_path.rglob("*"))
        if path.is_file()
    ]
    manifest = {
        "format": "prime-rl-packed-sft-v1",
        "seq_len": 4,
        "effective_tokenizer_json_sha256": hashlib.sha256(backend_payload.encode()).hexdigest(),
        "special_token_ids": {
            "bos_token_id": None,
            "eos_token_id": 68,
            "pad_token_id": 3,
            "unk_token_id": None,
        },
        "columns": {
            "input_ids": "uint32",
            "target_ids": "uint32",
            "loss_mask": "uint8",
            "seq_lens": "list<int64>",
        },
        "shards": [{"path": "chunk_00000", "rows": declared_rows, "files": files}],
        "stats": {"packed_rows": declared_rows},
    }
    manifest_bytes = json.dumps(manifest).encode()
    (root / "tokenized_shards.json").write_bytes(manifest_bytes)
    (root / "_READY").write_text(
        json.dumps(
            {
                "format": "prime-rl-packed-sft-v1",
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }
        )
    )
    return root, backend_payload


def test_load_pretokenized_dataset_validates_and_roundtrips(tmp_path):
    root, backend_payload = _write_pretokenized_artifact(tmp_path)
    tokenizer = _DummyPretokenizedTokenizer(backend_payload)
    config = PretokenizedSFTDataConfig(path=root, seq_len=4, shuffle=False)

    loaded = load_pretokenized_dataset(config, tokenizer)
    sample = next(iter(PretokenizedSFTDataset(loaded, shuffle=False, seq_len=4)))

    assert sample["input_ids"] == [10, 11, 20, 0]
    assert sample["target_ids"] == [11, 12, 21, 0]
    assert sample["loss_mask"] == [True, True, True, False]
    assert sample["seq_lens"] == [2, 1, 1]


def test_load_pretokenized_dataset_rejects_tokenizer_or_row_drift(tmp_path):
    root, _ = _write_pretokenized_artifact(tmp_path, declared_rows=2)
    config = PretokenizedSFTDataConfig(path=root, seq_len=4, shuffle=False)

    with pytest.raises(ValueError, match="tokenizer differ"):
        load_pretokenized_dataset(config, _DummyPretokenizedTokenizer("wrong"))
    with pytest.raises(ValueError, match="row count differs"):
        load_pretokenized_dataset(config, _DummyPretokenizedTokenizer('{"model":"test"}'))


def test_load_pretokenized_dataset_rejects_file_or_schema_drift(tmp_path):
    root, backend_payload = _write_pretokenized_artifact(tmp_path)
    config = PretokenizedSFTDataConfig(path=root, seq_len=4, shuffle=False)
    (root / "chunk_00000" / "unexpected").write_text("drift")

    with pytest.raises(ValueError, match="file sizes or SHA-256 hashes differ"):
        load_pretokenized_dataset(config, _DummyPretokenizedTokenizer(backend_payload))

    hash_root, hash_backend_payload = _write_pretokenized_artifact(tmp_path / "hash")
    hash_config = PretokenizedSFTDataConfig(path=hash_root, seq_len=4, shuffle=False)
    manifest = json.loads((hash_root / "tokenized_shards.json").read_text())
    declared_file = hash_root / "chunk_00000" / manifest["shards"][0]["files"][0]["path"]
    content = declared_file.read_bytes()
    declared_file.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(ValueError, match="SHA-256 hashes differ"):
        load_pretokenized_dataset(hash_config, _DummyPretokenizedTokenizer(hash_backend_payload))

    schema_root, schema_backend_payload = _write_pretokenized_artifact(tmp_path / "schema", loss_mask_dtype="int64")
    schema_config = PretokenizedSFTDataConfig(path=schema_root, seq_len=4, shuffle=False)
    with pytest.raises(ValueError, match="must contain uint8"):
        load_pretokenized_dataset(schema_config, _DummyPretokenizedTokenizer(schema_backend_payload))


def test_load_pretokenized_dataset_requires_manifest(tmp_path):
    root = tmp_path / "bare"
    Dataset.from_dict({"input_ids": [[1]]}).save_to_disk(root)
    config = PretokenizedSFTDataConfig(path=root, seq_len=1, shuffle=False)

    with pytest.raises(FileNotFoundError, match="Rebuild or migrate"):
        load_pretokenized_dataset(config, _DummyPretokenizedTokenizer("unused"))


def test_load_pretokenized_dataset_ready_marker_binds_manifest(tmp_path):
    root, backend_payload = _write_pretokenized_artifact(tmp_path)
    manifest_path = root / "tokenized_shards.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    config = PretokenizedSFTDataConfig(path=root, seq_len=4, shuffle=False)

    with pytest.raises(ValueError, match="does not match its ready marker"):
        load_pretokenized_dataset(config, _DummyPretokenizedTokenizer(backend_payload))


def test_pretokenized_shard_paths_cannot_escape_artifact_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)

    for shard_path in ("", str(outside), "../outside", "link"):
        with pytest.raises(ValueError, match="within the artifact root"):
            sft_data._resolve_shard_path(root, {"path": shard_path})


def test_distributed_shard_hash_verification_runs_only_on_rank_zero(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sft_data.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(sft_data.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(sft_data.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(sft_data.torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(sft_data.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        sft_data.torch.distributed,
        "broadcast_object_list",
        lambda result, src, device: None,
    )
    monkeypatch.setattr(sft_data, "_validate_shard_files", lambda path, files: calls.append(path))

    sft_data._validate_shard_files_once([(tmp_path, [])])
    assert calls == []

    monkeypatch.setattr(sft_data.torch.distributed, "get_rank", lambda: 0)
    sft_data._validate_shard_files_once([(tmp_path, [])])
    assert calls == [tmp_path]


def test_distributed_shard_hash_verification_propagates_rank_zero_error(monkeypatch, tmp_path):
    monkeypatch.setattr(sft_data.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(sft_data.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(sft_data.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(sft_data.torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(sft_data.torch.distributed, "get_rank", lambda: 1)

    def propagate_error(result, src, device):
        result[0] = "ValueError: corrupted shard"

    monkeypatch.setattr(sft_data.torch.distributed, "broadcast_object_list", propagate_error)
    with pytest.raises(RuntimeError, match="corrupted shard"):
        sft_data._validate_shard_files_once([(tmp_path, [])])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"input_ids": [1, 2], "target_ids": [2, 3], "loss_mask": [1, 1]},
            "missing required columns",
        ),
        (
            {
                "input_ids": [1, 2],
                "target_ids": [2, 3],
                "loss_mask": [1, 1],
                "seq_lens": [1, 2],
            },
            "sum to data.seq_len",
        ),
        (
            {
                "input_ids": [1, 2],
                "target_ids": [2, 3],
                "loss_mask": [0, 0],
                "seq_lens": [2],
            },
            "no trainable tokens",
        ),
    ],
)
def test_pretokenized_dataset_rejects_invalid_packs(row, message):
    dataset = PretokenizedSFTDataset(Dataset.from_list([row]), shuffle=False, seq_len=2)

    with pytest.raises(ValueError, match=message):
        next(iter(dataset))


def test_cat_dataset_preserves_nested_pretokenized_boundaries():
    sample = _sft_sample([1, 2, 3, 4])
    sample["seq_lens"] = [2, 1, 1]

    packed = next(iter(CatDataset([sample], seq_len=4)))

    assert packed["seq_lens"] == [2, 1, 1]


def test_cat_dataset_packs_multimodal_samples():
    dataset = CatDataset(
        [
            _sft_sample(
                [1, 2],
                mm_kwargs={
                    "pixel_values": torch.ones(2, 3),
                    "image_grid_thw": torch.tensor([[1, 1, 2]]),
                },
                mm_token_type_ids=[0, 1],
            ),
            _sft_sample(
                [3, 4, 5],
                mm_kwargs={
                    "pixel_values": 2 * torch.ones(3, 3),
                    "image_grid_thw": torch.tensor([[1, 1, 3]]),
                },
                mm_token_type_ids=[0, 1, 1],
            ),
        ],
        seq_len=5,
    )

    packed = next(iter(dataset))

    assert packed["input_ids"] == [1, 2, 3, 4, 5]
    assert packed["seq_lens"] == [2, 3]
    assert packed["mm_token_type_ids"] == [0, 1, 0, 1, 1]
    assert packed["mm_kwargs"]["pixel_values"].shape == (5, 3)
    assert packed["mm_kwargs"]["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 3]]


def test_cat_dataset_packs_text_and_multimodal_samples_together():
    dataset = CatDataset(
        [
            _sft_sample([1]),
            _sft_sample(
                [2, 3],
                mm_kwargs={
                    "pixel_values": torch.ones(2, 3),
                    "image_grid_thw": torch.tensor([[1, 1, 2]]),
                },
                mm_token_type_ids=[0, 1],
            ),
            _sft_sample([4]),
            _sft_sample([5, 6]),
        ],
        seq_len=5,
    )

    dataiter = iter(dataset)
    packed = next(dataiter)
    text_pack = next(dataiter)

    assert packed["input_ids"] == [1, 2, 3, 4, 0]
    assert packed["loss_mask"] == [True, True, True, True, False]
    assert packed["seq_lens"] == [1, 2, 2]
    assert packed["mm_kwargs"] is not None
    assert packed["mm_token_type_ids"] == [0, 0, 1, 0, 0]
    assert text_pack["input_ids"] == [5, 6, 0, 0, 0]
    assert text_pack["loss_mask"] == [True, True, False, False, False]
    assert text_pack["seq_lens"] == [5]
    assert text_pack["mm_kwargs"] is None
    assert text_pack["mm_token_type_ids"] is None


def test_pretokenized_shuffle_does_not_modify_saved_artifact(tmp_path):
    root, backend_payload = _write_pretokenized_artifact(tmp_path)
    tokenizer = _DummyPretokenizedTokenizer(backend_payload)
    config = PretokenizedSFTDataConfig(path=root, seq_len=4, shuffle=True)
    loaded = load_pretokenized_dataset(config, tokenizer)
    before = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file()}
    samples = list(PretokenizedSFTDataset(loaded, shuffle=True, seq_len=4, max_epochs=2))
    assert len(samples) == 2
    after = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert len(load_pretokenized_dataset(config, tokenizer)) == 1
