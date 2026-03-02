from typing import Any

from torch import nn
from transformers.integrations import accelerate as hf_accelerate


def revert_weight_conversion_if_supported(model: nn.Module, state_dict: dict[str, Any]) -> dict[str, Any]:
    if not (hasattr(hf_accelerate, "get_device") and hasattr(hf_accelerate, "offload_weight")):
        return state_dict
    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, state_dict)
