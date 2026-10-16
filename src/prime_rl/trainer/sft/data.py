import hashlib
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, interleave_datasets, load_dataset, load_from_disk
from jaxtyping import Bool, Int
from renderers.base import MultiModalData, PlaceholderRange, Renderer, build_training_sample
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.configs.sft import DataConfig, LossMaskConfig, PretokenizedSFTDataConfig, SFTDataConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from prime_rl.utils.logger import get_logger

TOKENIZED_SHARDS_MANIFEST_NAME = "tokenized_shards.json"


def requires_renderer(data_type: str, *, has_validation: bool) -> bool:
    return data_type == "sft" or has_validation


class Sample(TypedDict):
    input_ids: list[int]
    position_ids: list[int]
    loss_mask: list[bool]
    target_ids: list[int]
    seq_lens: list[int]
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: list[int] | None


class Batch(TypedDict):
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    target_ids: Int[Tensor, "batch seq"]
    loss_mask: Bool[Tensor, "batch seq"]
    seq_lens: Int[Tensor, "packed"]
    mm_kwargs: dict[str, Tensor] | None
    mm_token_type_ids: Int[Tensor, "batch seq"] | None


class StatefulIterableDataset(Stateful, IterableDataset):
    """SFT dataset are iterable (infinite) and stateful (can be checkpointed)."""

    def __init__(self, non_dp_size: int = 1):
        self.step, self.epoch = 0, 0
        self.num_samples = defaultdict(int)
        self.num_tokens = defaultdict(int)
        self.fast_forward = False
        self.non_dp_size = non_dp_size
        self._setup_world_info()

    def state_dict(self) -> dict:
        return {"step": self.step, "epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        assert "step" in state_dict and "epoch" in state_dict
        self.fast_forward = True
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]

    def _setup_world_info(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        world = get_world()
        assert world.world_size % self.non_dp_size == 0, "world_size must be divisible by non_dp_size"
        self.data_rank = world.rank // self.non_dp_size * num_workers + worker_id
        self.data_world_size = world.world_size // self.non_dp_size * num_workers


class FakeDataset(StatefulIterableDataset):
    """A dataset of fake tokens"""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        length: Literal["fixed", "variable"] = "fixed",
        input_ids: Literal["increasing", "random"] = "random",
        seed: int = 0,
        non_dp_size: int = 1,
    ):
        super().__init__(non_dp_size)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.input_ids = input_ids
        self.seed = seed

    def _draw_sample(self, generator: torch.Generator) -> tuple[int, list[int] | None]:
        # Consume this samples "randomness" - fast forwarding must replay it to restore the generator state
        seq_len = (
            int(torch.randint(1, self.seq_len, (1,), generator=generator).item())
            if self.length == "variable"
            else self.seq_len
        )
        random_input_ids = (
            torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=generator).long().tolist()
            if self.input_ids == "random"
            else None
        )
        return seq_len, random_input_ids

    def __iter__(self):
        self._setup_world_info()
        # use a rank seeded PRNG instead of torch global default PRNG because with num workers > 0
        # the data loader reseeds the global PRNG per worker process
        generator = torch.Generator().manual_seed(self.seed + self.data_rank)
        if self.fast_forward:
            # step counts globally emmited samples but this rank is only emitted every data_world_size-TH
            already_emitted = len(range(self.data_rank, self.step, self.data_world_size))
            for _ in range(already_emitted):
                self._draw_sample(generator)
            self.fast_forward = False

        while True:
            self.step += 1

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            seq_len, random_input_ids = self._draw_sample(generator)
            input_ids = [self.step - 1] * (seq_len + 1) if random_input_ids is None else random_input_ids
            position_ids = list(range(seq_len))
            loss_mask = [True] * seq_len
            fake_sample = {
                "input_ids": input_ids[:-1],
                "target_ids": input_ids[1:],
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "seq_lens": [seq_len],
                "mm_kwargs": None,
                "mm_token_type_ids": None,
            }
            self.num_samples["fake"] += 1
            self.num_tokens["fake"] += len(input_ids)
            yield fake_sample


def _flatten_mm_items(mm_items: dict[str, list[dict[str, Any]]]) -> dict[str, Tensor]:
    """Fold per-item renderer outputs into model-forward tensors."""
    out: dict[str, Tensor] = {}
    for items in mm_items.values():
        for item in items:
            for key, value in item.items():
                if not isinstance(value, (np.ndarray, Tensor)):
                    continue
                tensor = torch.as_tensor(value)
                out[key] = torch.cat([out[key], tensor], dim=0) if key in out else tensor
    return out


def _drop_null_fields(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Recursively strip ``None``-valued keys from dict structures.

    PyArrow's JSON loader unifies schemas across rows, so heterogeneous
    OAI content blocks (text vs image_url) end up with all union keys
    filled with ``None`` where absent. That confuses permissive
    content-type predicates inside renderers (e.g. ``"image_url" in item``
    returns ``True`` even when the value is null). Strip the noise before
    handing messages off to the renderer. Tool-call arguments are opaque
    JSON payloads, so preserve their null values.
    """
    if path[-3:] == ("tool_calls", "function", "arguments"):
        return value
    if isinstance(value, dict):
        return {k: _drop_null_fields(v, (*path, k)) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_null_fields(v, path) for v in value]
    return value


def _find_image_safe_cut(budget: int, mm: MultiModalData | None) -> int:
    """Return the largest cut at most ``budget`` outside placeholder runs."""
    if mm is None or not mm.mm_placeholders:
        return budget
    cut = budget
    for ranges in mm.mm_placeholders.values():
        for placeholder in ranges:
            if placeholder.offset < cut < placeholder.offset + placeholder.length:
                cut = placeholder.offset
    return cut


def _truncate_mm_data(mm: MultiModalData, cut: int) -> MultiModalData:
    """Drop multimodal items whose placeholder ranges extend past ``cut``."""
    new_placeholders: dict[str, list[PlaceholderRange]] = {}
    new_items: dict[str, list[dict[str, Any]]] = {}
    new_hashes: dict[str, list[str]] = {}
    for content_type, ranges in mm.mm_placeholders.items():
        keep = [index for index, placeholder in enumerate(ranges) if placeholder.offset + placeholder.length <= cut]
        if not keep:
            continue
        new_placeholders[content_type] = [ranges[index] for index in keep]
        new_items[content_type] = [mm.mm_items[content_type][index] for index in keep]
        if content_type in mm.mm_hashes:
            new_hashes[content_type] = [mm.mm_hashes[content_type][index] for index in keep]
    return MultiModalData(mm_hashes=new_hashes, mm_placeholders=new_placeholders, mm_items=new_items)


class SFTDataset(StatefulIterableDataset):
    """A dataset wrapping a HF SFT dataset with prompt/completion or raw messages format."""

    def __init__(
        self,
        dataset: Dataset,
        renderer: Renderer,
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        loss_mask_config: LossMaskConfig = LossMaskConfig(),
        max_examples: int | None = None,
        max_epochs: int | None = None,
        multimodal: bool = False,
    ):
        super().__init__(non_dp_size)
        self.logger = get_logger()
        self.dataset = dataset
        self.num_examples = len(self.dataset)
        self.renderer = renderer
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.loss_mask_config = loss_mask_config
        self.max_examples = max_examples
        self.max_epochs = max_epochs
        self.multimodal = multimodal

        # If specified, select a subset of the dataset
        if self.max_examples is not None:
            self.num_examples = min(self.num_examples, self.max_examples)
            self.dataset = self.dataset.take(self.max_examples)

    def _process(self, example: dict) -> dict | None:
        def resolve_messages(example: dict) -> list[dict]:
            # `messages` takes precedence over explicit split fields and is interpreted
            # as a whole-chat training sample with an empty prompt. Null-check rather
            # than key-check: Arrow schema union adds `messages: null` to
            # prompt/completion rows whenever other rows have a `messages` column.
            if example.get("messages") is not None:
                messages = normalize_messages(example["messages"], default_role="assistant")
            elif example.get("prompt") is not None and example.get("completion") is not None:
                messages = normalize_messages(example["prompt"], default_role="user") + normalize_messages(
                    example["completion"], default_role="assistant"
                )
            else:
                raise ValueError(
                    "All examples in the dataset must have either a 'messages' column "
                    "or both 'prompt' and 'completion' columns for SFT"
                )

            # Strip nulls before deserializing so genuine nulls inside tool-call
            # argument strings survive.
            messages = [_drop_null_fields(m) for m in messages]
            return deserialize_tool_calls(messages)

        messages = resolve_messages(example)

        # Parse available tools, if present - assumes OAI format. Accepts either
        # `tools` or `tool_defs` (the verifiers rollout format), as either a
        # JSON-encoded string of a list or a list of dicts; verifiers-shaped
        # tools are converted to OAI form for the chat template.
        raw_tools = example.get("tools", example.get("tool_defs"))
        if not raw_tools:
            tools = []
        else:
            if isinstance(raw_tools, str):
                raw_tools = json.loads(raw_tools)
            tools = [
                t
                if isinstance(t, dict) and t.get("type") == "function" and "function" in t
                else {
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description"),
                        "parameters": t.get("parameters"),
                        **({} if t.get("strict") is None else {"strict": t["strict"]}),
                    },
                }
                for t in raw_tools
            ]

        def should_mask(message: dict) -> bool:
            assert "role" in message, "Message must have a role"
            match message["role"]:
                case "user":
                    return self.loss_mask_config.user
                case "assistant":
                    return self.loss_mask_config.assistant
                case "system":
                    return self.loss_mask_config.system
                case "tool":
                    return self.loss_mask_config.tool
                case _:
                    raise ValueError(f"Invalid message role: {message['role']}")

        # Defer to the renderer's sampled_mask by default: a role filter would
        # drop sampled stop markers attributed to the next message (e.g. GLM's
        # turn-closing <|user|> / <|observation|>).
        role_to_mask = None if self.loss_mask_config.assistant else should_mask

        # Non-assistant roles are opted into the loss via the renderer's
        # body-only path: the message content is trained, not the role
        # scaffolding (e.g. <|im_start|>assistant) the harness emits.
        content_sft_roles = {role for role in ("user", "system", "tool") if getattr(self.loss_mask_config, role)}
        sample = build_training_sample(
            self.renderer,
            messages,
            role_to_mask=role_to_mask,
            tools=tools,
            content_sft_roles=content_sft_roles or None,
            ensure_final_stop=True,
        )
        input_ids = list(sample.token_ids)
        loss_mask = list(sample.loss_mask)
        mm = sample.multi_modal_data
        mm_token_type_ids = list(sample.mm_token_type_ids) if sample.mm_token_type_ids is not None else None
        if mm is not None and mm.mm_items and not self.multimodal:
            raise ValueError(
                "Renderer produced multimodal data but [model.vlm] is not set. "
                "Set [model.vlm] to train on multimodal samples."
            )

        # Causal shift: model predicts next token from current.
        target_ids = input_ids[1:]
        loss_mask = loss_mask[1:]
        input_ids = input_ids[:-1]
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[:-1]

        was_mm_truncated = False
        if mm is not None and len(input_ids) > self.seq_len:
            was_mm_truncated = True
            cut = _find_image_safe_cut(self.seq_len, mm)
            self.logger.debug(
                f"Truncating example {example.get('__index', '')} from "
                f"{len(input_ids)} → {cut} tokens (budget={self.seq_len})"
            )
            input_ids = input_ids[:cut]
            target_ids = target_ids[:cut]
            loss_mask = loss_mask[:cut]
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids[:cut]
            if mm.mm_items:
                mm = _truncate_mm_data(mm, cut)

        if was_mm_truncated and not set(self.renderer.get_stop_token_ids()) & set(target_ids):
            return None

        if sum(loss_mask[: self.seq_len]) == 0:
            self.logger.warning(
                f"Skipping example {example.get('__index', '')} because no trainable tokens were found within the context window ({self.seq_len}). This is to prevent NaN loss."
            )
            return None

        assert len(input_ids) == len(loss_mask) == len(target_ids), (
            f"input_ids, loss_mask and target_ids must have the same length, but got {len(input_ids)=}, {len(loss_mask)=}, {len(target_ids)=}"
        )
        assert sum(loss_mask) > 0, "There are no tokens in this sample that contribute to the loss"
        assert set(self.renderer.get_stop_token_ids()) & set(target_ids), (
            "A renderer stop token must be present in target_ids"
        )

        mm_kwargs: dict[str, Tensor] | None = None
        if mm is not None and mm.mm_items:
            mm_kwargs = _flatten_mm_items(mm.mm_items)
            if any("video" in key for key in mm_kwargs):
                raise ValueError("Video SFT is not supported; sample contains video inputs")
        if mm_token_type_ids is not None:
            assert len(mm_token_type_ids) == len(input_ids)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": list(range(len(input_ids))),
            "seq_lens": [len(input_ids)],
            "mm_kwargs": mm_kwargs,
            "mm_token_type_ids": mm_token_type_ids,
        }

    def __iter__(self):
        self._setup_world_info()
        dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset
        while True:
            self.step += 1

            # Determine epoch from current step
            epoch = (self.step - 1) // self.num_examples

            # Break if max epochs is reached
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break

            # Update stored epoch if new epoch is reached, optionally shuffle
            if epoch > self.epoch:
                self.epoch = epoch
                dataset = self.dataset.shuffle(seed=self.epoch + self.seed) if self.shuffle else self.dataset

            # Skip samples that don't belong to this data rank
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            # Get example
            example = dataset[(self.step - 1) % self.num_examples]

            # Process example
            processed_example = self._process(cast(dict, example))

            # If processed example is None, skip it (e.g. if tokenized sample exceeds context window)
            if processed_example is None:
                continue

            # Yield the example
            example = cast(dict, example)
            subset_or_split = example.get("__subset") or example.get("__split")
            self.logger.debug(
                f"Yield example {example.get('__index', '')}"
                + (f" from {subset_or_split} " if subset_or_split else " ")
                + f"with {len(processed_example.get('input_ids', []))} tokens ({sum(processed_example.get('loss_mask', []))} trainable tokens)"
            )
            self.num_samples[subset_or_split] += 1
            self.num_tokens[subset_or_split] += len(processed_example.get("input_ids", []))
            yield processed_example


class PretokenizedSFTDataset(StatefulIterableDataset):
    """Wrap pretokenized rows as PRIME-RL causal language-model samples."""

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        seq_len: int = 128,
        non_dp_size: int = 1,
        input_ids_column: str = "input_ids",
        target_ids_column: str = "target_ids",
        loss_mask_column: str = "loss_mask",
        seq_lens_column: str = "seq_lens",
        max_epochs: int | None = None,
    ):
        super().__init__()
        self.logger = get_logger()
        self.dataset = dataset
        self.num_examples = len(dataset)
        self.shuffle = shuffle
        self.seed = seed
        self.seq_len = seq_len
        self.input_ids_column = input_ids_column
        self.target_ids_column = target_ids_column
        self.loss_mask_column = loss_mask_column
        self.seq_lens_column = seq_lens_column
        self.max_epochs = max_epochs

        worker_info = get_worker_info()
        worker_id, num_workers = 0, 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        assert get_world().world_size % non_dp_size == 0, "world_size must be divisible by non_dp_size"
        self.data_rank = get_world().rank // non_dp_size * num_workers + worker_id
        self.data_world_size = get_world().world_size // non_dp_size * num_workers

    @staticmethod
    def _to_int_list(value: Any, column: str) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            raise ValueError(f"Column '{column}' must contain a list of token IDs")
        return [int(item) for item in value]

    @staticmethod
    def _to_bool_list(value: Any, column: str) -> list[bool]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            raise ValueError(f"Column '{column}' must contain a list of mask values")
        return [bool(item) for item in value]

    def _process(self, example: dict) -> Sample:
        required_columns = (
            self.input_ids_column,
            self.target_ids_column,
            self.loss_mask_column,
            self.seq_lens_column,
        )
        missing = [column for column in required_columns if column not in example]
        if missing:
            raise ValueError(f"Pretokenized dataset is missing required columns: {missing}")

        input_ids = self._to_int_list(example[self.input_ids_column], self.input_ids_column)
        target_ids = self._to_int_list(example[self.target_ids_column], self.target_ids_column)
        loss_mask = self._to_bool_list(example[self.loss_mask_column], self.loss_mask_column)
        seq_lens = self._to_int_list(example[self.seq_lens_column], self.seq_lens_column)

        lengths = {len(input_ids), len(target_ids), len(loss_mask)}
        if lengths != {self.seq_len}:
            raise ValueError(
                "Pretokenized rows must already be packed to data.seq_len: "
                f"input={len(input_ids)}, target={len(target_ids)}, mask={len(loss_mask)}, expected={self.seq_len}"
            )
        if not seq_lens or any(length <= 0 for length in seq_lens) or sum(seq_lens) != self.seq_len:
            raise ValueError(
                "Packed seq_lens must be non-empty, positive, and sum to data.seq_len: "
                f"seq_lens={seq_lens}, expected={self.seq_len}"
            )

        if not any(loss_mask):
            raise ValueError(
                "Pretokenized row has no trainable tokens and violates the packed artifact contract: "
                f"example={example.get('__index', '')!r}, seq_len={self.seq_len}"
            )
        position_ids = [position for length in seq_lens for position in range(length)]
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "seq_lens": seq_lens,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
        }

    def __iter__(self):
        dataset = self.dataset.shuffle(seed=self.epoch + self.seed, keep_in_memory=True) if self.shuffle else self.dataset
        while True:
            self.step += 1
            epoch = (self.step - 1) // self.num_examples
            if self.max_epochs is not None and epoch >= self.max_epochs:
                break
            if epoch > self.epoch:
                self.epoch = epoch
                dataset = self.dataset.shuffle(seed=self.epoch + self.seed, keep_in_memory=True) if self.shuffle else self.dataset
            if (self.step - 1) % self.data_world_size != self.data_rank:
                continue

            example = cast(dict, dataset[(self.step - 1) % self.num_examples])
            processed_example = self._process(example)
            self.num_samples["pretokenized"] += 1
            self.num_tokens["pretokenized"] += len(processed_example["input_ids"])
            yield processed_example


class CatDataset(StatefulIterableDataset):
    """Concatenate text and multimodal samples into one fixed-length row."""

    def __init__(self, dataset: StatefulIterableDataset, seq_len: int):
        self.logger = get_logger()
        self.dataset = dataset
        self.seq_len = seq_len
        self.pending_sample: Sample | None = None

    def state_dict(self) -> dict:
        state = {
            "dataset": self.dataset.state_dict(),
            "progress": {
                "num_samples": dict(self.dataset.num_samples),
                "num_tokens": dict(self.dataset.num_tokens),
            },
        }
        if self.pending_sample is not None:
            state["pending_sample"] = self.pending_sample
        return state

    def load_state_dict(self, state_dict: dict):
        self.dataset.load_state_dict(state_dict["dataset"])
        progress = state_dict.get("progress", {})
        self.dataset.num_samples.update(progress.get("num_samples", {}))
        self.dataset.num_tokens.update(progress.get("num_tokens", {}))
        self.pending_sample = state_dict.get("pending_sample")

    def __iter__(self):
        packed_samples = defaultdict(list)
        packed_samples["mm_kwargs"] = None
        packed_samples["mm_token_type_ids"] = None
        seq_len = 0

        pending_sample = self.pending_sample
        self.pending_sample = None

        def samples():
            if pending_sample is not None:
                yield pending_sample
            yield from self.dataset

        for sample in samples():
            sample_len = len(sample["input_ids"])
            would_overflow = seq_len + sample_len > self.seq_len
            if seq_len > 0 and would_overflow:
                self.pending_sample = sample
                yield self._finalize_pack(packed_samples, self.seq_len)
                self.pending_sample = None
                packed_samples = defaultdict(list)
                packed_samples["mm_kwargs"] = None
                packed_samples["mm_token_type_ids"] = None
                seq_len = 0

            existing_len = len(packed_samples["input_ids"])
            for key in ("input_ids", "position_ids", "loss_mask", "target_ids"):
                value = sample[key]
                assert isinstance(value, list)
                packed_samples[key].extend(value)
            packed_samples["seq_lens"].extend(sample["seq_lens"])

            sample_mm_kwargs = sample.get("mm_kwargs")
            sample_mm_type_ids = sample.get("mm_token_type_ids")
            if sample_mm_kwargs is None:
                if packed_samples["mm_token_type_ids"] is not None:
                    packed_samples["mm_token_type_ids"].extend([0] * sample_len)
            else:
                if packed_samples["mm_kwargs"] is not None and (
                    (packed_samples["mm_token_type_ids"] is None) != (sample_mm_type_ids is None)
                ):
                    raise ValueError("Cannot pack multimodal samples with mixed mm_token_type_ids")

                if packed_samples["mm_kwargs"] is None:
                    packed_samples["mm_kwargs"] = dict(sample_mm_kwargs)
                else:
                    if packed_samples["mm_kwargs"].keys() != sample_mm_kwargs.keys():
                        raise ValueError("Cannot pack multimodal samples with different mm_kwargs keys")
                    for key, value in sample_mm_kwargs.items():
                        packed_samples["mm_kwargs"][key] = torch.cat([packed_samples["mm_kwargs"][key], value], dim=0)

                if packed_samples["mm_token_type_ids"] is None and sample_mm_type_ids is not None:
                    packed_samples["mm_token_type_ids"] = [0] * existing_len
                if packed_samples["mm_token_type_ids"] is not None:
                    packed_samples["mm_token_type_ids"].extend(sample_mm_type_ids or [0] * sample_len)

            seq_len += sample_len

            if seq_len >= self.seq_len:
                yield self._finalize_pack(packed_samples, self.seq_len)
                packed_samples = defaultdict(list)
                packed_samples["mm_kwargs"] = None
                packed_samples["mm_token_type_ids"] = None
                seq_len = 0

        if seq_len > 0:
            yield self._finalize_pack(packed_samples, self.seq_len)

    def _finalize_pack(self, packed: dict[str, Any], seq_len: int) -> dict:
        result: dict[str, Any] = {
            k: packed[k][:seq_len] for k in ("input_ids", "position_ids", "loss_mask", "target_ids")
        }
        result["seq_lens"] = []
        remaining = len(result["input_ids"])
        for sample_len in packed["seq_lens"]:
            if remaining <= 0:
                break
            kept = min(sample_len, remaining)
            if kept > 0:
                result["seq_lens"].append(kept)
            remaining -= kept
        pad_len = seq_len - len(result["input_ids"])
        if pad_len > 0:
            result["input_ids"].extend([0] * pad_len)
            result["position_ids"].extend(range(pad_len))
            result["loss_mask"].extend([False] * pad_len)
            result["target_ids"].extend([0] * pad_len)
            result["seq_lens"][-1] += pad_len
        result["mm_kwargs"] = packed["mm_kwargs"]
        if packed["mm_token_type_ids"] is not None:
            result["mm_token_type_ids"] = packed["mm_token_type_ids"][:seq_len] + [0] * pad_len
        else:
            result["mm_token_type_ids"] = None
        return result


def cat_collate(samples: list[Sample]) -> Batch:
    # CPU tensors only: this runs in dataloader workers then the trainer moves batches to the GPU with async copies from pinned memory
    (sample,) = samples
    mm_kwargs = sample.get("mm_kwargs")
    mm_token_type_ids = sample.get("mm_token_type_ids")
    return {
        "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0),
        "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long).unsqueeze(0),
        "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.bool).unsqueeze(0),
        "target_ids": torch.tensor(sample["target_ids"], dtype=torch.long).unsqueeze(0),
        "seq_lens": torch.tensor(sample["seq_lens"], dtype=torch.long),
        "mm_kwargs": dict(mm_kwargs) if mm_kwargs is not None else None,
        "mm_token_type_ids": (
            torch.tensor(mm_token_type_ids, dtype=torch.long).unsqueeze(0) if mm_token_type_ids is not None else None
        ),
    }


def setup_and_interleave_datasets(
    dataset_name: str,
    subsets_and_splits: list[tuple[str | None, str]],
    probabilities: list[float] | None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"],
    seed: int = 0,
) -> Dataset:
    logger = get_logger()
    datasets = []
    for subset, split in subsets_and_splits:
        logger.debug(f"Loading dataset {dataset_name} with {subset=} and {split=}")
        dataset = cast(Dataset, load_dataset(dataset_name, subset, split=split))
        num_examples = len(dataset)
        dataset = dataset.add_column("__subset", [subset] * num_examples, new_fingerprint=str(uuid.uuid4()))
        dataset = dataset.add_column("__split", [split] * num_examples, new_fingerprint=str(uuid.uuid4()))
        dataset = dataset.add_column("__index", list(range(num_examples)), new_fingerprint=str(uuid.uuid4()))
        datasets.append(dataset)
    if len(datasets) > 1:
        logger.debug(f"Interleaving datasets with {probabilities=} and {stopping_strategy=}")
        dataset = interleave_datasets(
            datasets,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
            seed=seed,
        )
    else:
        dataset = datasets[0]

    return dataset


def load_sft_dataset(config: SFTDataConfig) -> Dataset:
    """Load and interleave the raw HF dataset. This is the expensive I/O step."""
    logger = get_logger()
    if config.subsets is None and config.splits is None:
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, "train")],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    elif config.subsets is not None and config.splits is None:
        logger.debug(f"Loading datasets for subsets {config.subsets} with default split 'train'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(subset, "train") for subset in config.subsets],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    elif config.subsets is None and config.splits is not None:
        logger.debug(f"Loading datasets for splits {config.splits} with default subset 'None'")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=[(None, split) for split in config.splits],
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )
    else:
        assert config.subsets is not None and config.splits is not None
        logger.debug(f"Loading datasets for subsets {config.subsets} with splits {config.splits}")
        return setup_and_interleave_datasets(
            dataset_name=config.name,
            subsets_and_splits=list(zip(config.subsets, config.splits)),
            probabilities=config.probabilities,
            stopping_strategy=config.stopping_strategy,
        )


def _resolve_shard_path(root: Path, shard: dict) -> Path:
    if not isinstance(shard.get("path"), str):
        raise ValueError(f"Invalid shard entry in {TOKENIZED_SHARDS_MANIFEST_NAME}: {shard!r}")
    relative_path = Path(shard["path"])
    if not relative_path.parts or relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"Pretokenized shard paths must be relative and stay within the artifact root: {relative_path}"
        )
    resolved_root = root.resolve()
    resolved_path = (resolved_root / relative_path).resolve()
    if not resolved_path.is_relative_to(resolved_root) or resolved_path == resolved_root:
        raise ValueError(f"Pretokenized shard paths must resolve within the artifact root: {relative_path}")
    return resolved_path


def _effective_tokenizer_json_sha256(tokenizer: PreTrainedTokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        raise ValueError(
            "Pretokenized SFT requires a fast tokenizer with backend_tokenizer.to_str() "
            "so packed token IDs can be verified"
        )
    return hashlib.sha256(to_str().encode("utf-8")).hexdigest()


def _manifest_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _validate_pretokenized_features(
    dataset: Dataset,
    config: PretokenizedSFTDataConfig,
    *,
    shard_path: Path,
) -> None:
    expected = {
        config.input_ids_column: ("uint32", config.seq_len),
        config.target_ids_column: ("uint32", config.seq_len),
        config.loss_mask_column: ("uint8", config.seq_len),
        config.seq_lens_column: ("int64", None),
    }
    missing = [name for name in expected if name not in dataset.column_names]
    if missing:
        raise ValueError(f"Pretokenized shard {shard_path} is missing columns: {missing}")

    for column, (expected_dtype, expected_length) in expected.items():
        feature = dataset.features[column]
        item_feature = getattr(feature, "feature", None)
        actual_dtype = getattr(item_feature, "dtype", None)
        actual_length = getattr(feature, "length", None)
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"Pretokenized shard {shard_path} column {column!r} must contain {expected_dtype}, got {actual_dtype!r}"
            )
        if expected_length is not None and actual_length != expected_length:
            raise ValueError(
                f"Pretokenized shard {shard_path} column {column!r} must have fixed "
                f"length {expected_length}, got {actual_length!r}"
            )


def _validate_shard_files(shard_path: Path, files: Any) -> None:
    if not isinstance(files, list) or not files:
        raise ValueError(f"Pretokenized shard {shard_path} has no declared files")
    declared: dict[str, tuple[int, str]] = {}
    resolved_shard_path = shard_path.resolve()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"Invalid file entry for pretokenized shard {shard_path}: {entry!r}")
        relative_path = Path(entry["path"])
        if not relative_path.parts or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Pretokenized shard file paths must stay within {shard_path}: {relative_path}")
        size_bytes = _manifest_positive_int(entry.get("size_bytes"), f"file size for {shard_path / relative_path}")
        sha256 = entry.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError(f"Invalid SHA-256 for pretokenized shard file {shard_path / relative_path}")
        key = relative_path.as_posix()
        if key in declared:
            raise ValueError(f"Duplicate file entry for pretokenized shard {shard_path}: {key}")
        declared_path = (resolved_shard_path / relative_path).resolve()
        if not declared_path.is_relative_to(resolved_shard_path):
            raise ValueError(f"Pretokenized shard file paths must resolve within {shard_path}: {relative_path}")
        declared[key] = (size_bytes, sha256)

    actual: dict[str, tuple[int, str]] = {}
    for path in shard_path.rglob("*"):
        if not path.is_file():
            continue
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(resolved_shard_path):
            raise ValueError(f"Pretokenized shard contains a file outside its directory: {path}")
        actual[path.relative_to(shard_path).as_posix()] = (path.stat().st_size, _sha256_file(path))
    if actual != declared:
        raise ValueError(
            f"Pretokenized shard file sizes or SHA-256 hashes differ from the manifest for {shard_path}: "
            f"declared={declared}, actual={actual}"
        )


def _validate_shard_files_once(shards: list[tuple[Path, Any]]) -> None:
    dist = torch.distributed
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if not distributed:
        for shard_path, files in shards:
            _validate_shard_files(shard_path, files)
        return

    result: list[str | None] = [None]
    if dist.get_rank() == 0:
        try:
            for shard_path, files in shards:
                _validate_shard_files(shard_path, files)
        except Exception as exc:
            result[0] = f"{type(exc).__name__}: {exc}"

    backend = str(dist.get_backend()).lower()
    device = torch.device("cuda", torch.cuda.current_device()) if "nccl" in backend else None
    dist.broadcast_object_list(result, src=0, device=device)
    if result[0] is not None:
        raise RuntimeError(f"Rank-zero pretokenized shard verification failed: {result[0]}")


def load_pretokenized_dataset(
    config: PretokenizedSFTDataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """Load and verify packed Dataset shards listed by an artifact manifest."""
    root = config.path.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Pretokenized dataset path does not exist: {root}")

    manifest_path = root / TOKENIZED_SHARDS_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Pretokenized dataset is missing its verification manifest: {manifest_path}. "
            "Rebuild or migrate the artifact to the packed manifest contract before training."
        )
    ready_path = root / "_READY"
    if not ready_path.exists():
        raise ValueError(f"Pretokenized artifact is incomplete because {root / '_READY'} is missing")
    manifest_bytes = manifest_path.read_bytes()
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pretokenized artifact has an invalid ready marker: {ready_path}") from exc
    if not isinstance(ready, dict) or ready.get("format") != "prime-rl-packed-sft-v1":
        raise ValueError(f"Pretokenized artifact has an invalid ready marker: {ready_path}")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if ready.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Pretokenized artifact manifest does not match its ready marker: {manifest_path}")
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    if manifest.get("format") != "prime-rl-packed-sft-v1":
        raise ValueError(
            f"{manifest_path} must declare format='prime-rl-packed-sft-v1', got {manifest.get('format')!r}"
        )
    if manifest.get("seq_len") != config.seq_len:
        raise ValueError(
            f"{manifest_path} was packed at seq_len={manifest.get('seq_len')}, "
            f"but the training config requests seq_len={config.seq_len}"
        )
    expected_columns = {
        config.input_ids_column: "uint32",
        config.target_ids_column: "uint32",
        config.loss_mask_column: "uint8",
        config.seq_lens_column: "list<int64>",
    }
    if manifest.get("columns") != expected_columns:
        raise ValueError(
            f"{manifest_path} columns do not match the training config: "
            f"expected={expected_columns}, got={manifest.get('columns')!r}"
        )
    packed_tokenizer_sha256 = manifest.get("effective_tokenizer_json_sha256")
    runtime_tokenizer_sha256 = _effective_tokenizer_json_sha256(tokenizer)
    if packed_tokenizer_sha256 != runtime_tokenizer_sha256:
        raise ValueError(
            "Pretokenized artifact and runtime tokenizer differ: "
            f"packed={packed_tokenizer_sha256!r}, runtime={runtime_tokenizer_sha256!r}"
        )
    special_token_ids = manifest.get("special_token_ids")
    if not isinstance(special_token_ids, dict):
        raise ValueError(f"{manifest_path} must contain special_token_ids")
    for token_name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if token_name not in special_token_ids:
            raise ValueError(f"{manifest_path} special_token_ids is missing {token_name}")
        packed_token_id = special_token_ids.get(token_name)
        runtime_token_id = getattr(tokenizer, token_name, None)
        if packed_token_id != runtime_token_id:
            raise ValueError(
                f"Pretokenized artifact {token_name}={packed_token_id!r} does not match "
                f"runtime tokenizer {token_name}={runtime_token_id!r}"
            )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"{manifest_path} must contain a non-empty 'shards' list")
    resolved_shards: list[tuple[Path, int, Any]] = []
    resolved_paths: set[Path] = set()
    declared_rows = 0
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"Invalid shard entry in {manifest_path}: {shard!r}")
        shard_rows = _manifest_positive_int(shard.get("rows"), f"shards[{shard_index}].rows")
        shard_path = _resolve_shard_path(root, shard).resolve()
        if shard_path in resolved_paths:
            raise ValueError(f"Duplicate pretokenized shard path: {shard_path}")
        resolved_paths.add(shard_path)
        resolved_shards.append((shard_path, shard_rows, shard.get("files")))
        declared_rows += shard_rows

    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        raise ValueError(f"{manifest_path} must contain aggregate stats")
    packed_rows = _manifest_positive_int(stats.get("packed_rows"), "stats.packed_rows")
    if packed_rows != declared_rows:
        raise ValueError(
            f"{manifest_path} stats.packed_rows={packed_rows} does not match the shard total {declared_rows}"
        )

    _validate_shard_files_once([(shard_path, files) for shard_path, _, files in resolved_shards])
    datasets: list[Dataset] = []
    for shard_path, shard_rows, _ in resolved_shards:
        if not shard_path.is_dir():
            raise FileNotFoundError(f"Pretokenized shard path does not exist: {shard_path}")
        loaded = load_from_disk(str(shard_path))
        if not isinstance(loaded, Dataset):
            raise TypeError(f"Pretokenized shard must be a Dataset, not a DatasetDict: {shard_path}")
        if len(loaded) != shard_rows:
            raise ValueError(
                f"Pretokenized shard row count differs from manifest for {shard_path}: "
                f"manifest={shard_rows}, dataset={len(loaded)}"
            )
        _validate_pretokenized_features(loaded, config, shard_path=shard_path)
        datasets.append(loaded)
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def setup_dataset(
    tokenizer: PreTrainedTokenizer,
    config: DataConfig,
    non_dp_size: int = 1,
    *,
    max_epochs: int | None = None,
    raw_dataset: Dataset | None = None,
    renderer: Renderer | None = None,
    multimodal: bool = False,
) -> StatefulIterableDataset:
    if config.type == "fake":
        return FakeDataset(
            vocab_size=tokenizer.vocab_size,
            seq_len=config.seq_len,
            length=config.length,
            input_ids=config.input_ids,
            seed=config.seed,
            non_dp_size=non_dp_size,
        )
    elif config.type == "sft":
        if renderer is None:
            raise ValueError("SFT data requires a renderer.")
        if raw_dataset is None:
            raw_dataset = load_sft_dataset(config)
        return SFTDataset(
            raw_dataset,
            renderer,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            loss_mask_config=config.loss_mask,
            non_dp_size=non_dp_size,
            max_epochs=max_epochs,
            multimodal=multimodal,
        )
    elif config.type == "pretokenized":
        if raw_dataset is None:
            raw_dataset = load_pretokenized_dataset(config, tokenizer)
        return PretokenizedSFTDataset(
            raw_dataset,
            shuffle=config.shuffle,
            seed=config.seed,
            seq_len=config.seq_len,
            non_dp_size=non_dp_size,
            input_ids_column=config.input_ids_column,
            target_ids_column=config.target_ids_column,
            loss_mask_column=config.loss_mask_column,
            seq_lens_column=config.seq_lens_column,
            max_epochs=max_epochs,
        )
    else:
        raise ValueError(f"Invalid dataset type: {config.type}")


def setup_dataloader(dataset: StatefulIterableDataset, config: DataConfig) -> StatefulDataLoader:
    packing_dataset = CatDataset(dataset, config.seq_len * config.micro_batch_size)
    return StatefulDataLoader(
        packing_dataset,
        batch_size=1,
        collate_fn=cat_collate,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def get_dataset_state(dataloader: StatefulDataLoader) -> dict:
    """Dataset position per worker for the startup log, parsed from ``StatefulDataLoader.state_dict()``.

    The loader is the only source that is correct in every case: after a resume's
    ``load_state_dict`` the restored position exists solely in the loader's stashed
    state (it reaches the dataset copies inside workers when the iterator forks them;
    the main-process dataset object stays at position zero). The keys are torchdata's
    private worker-snapshot layout."""
    snapshots = dataloader.state_dict()["_snapshot"]["_worker_snapshots"]
    return {wid: snap["dataset_state"]["dataset"] for wid, snap in sorted(snapshots.items())}


def get_dataset_progress(dataloader: StatefulDataLoader) -> dict:
    """Dataset position and aggregate counters from dataloader workers."""
    snapshot = dataloader.state_dict()["_snapshot"]
    worker_snapshots = snapshot["_worker_snapshots"]
    positions = [worker_snapshot["dataset_state"]["dataset"] for worker_snapshot in worker_snapshots.values()]
    furthest = max(positions, key=lambda position: position["step"])
    num_samples = defaultdict(int)
    num_tokens = defaultdict(int)
    for worker_snapshot in worker_snapshots.values():
        progress = worker_snapshot["dataset_state"].get("progress", {})
        for name, count in progress.get("num_samples", {}).items():
            num_samples[name] += count
        for name, count in progress.get("num_tokens", {}).items():
            num_tokens[name] += count
    return {
        **furthest,
        "num_samples": dict(num_samples),
        "num_tokens": dict(num_tokens),
    }
