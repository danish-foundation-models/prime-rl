from transformers.integrations import accelerate as hf_accelerate


def revert_weight_conversion_if_supported(model, state_dict):
    if not (hasattr(hf_accelerate, "get_device") and hasattr(hf_accelerate, "offload_weight")):
        return state_dict
    try:
        from transformers.core_model_loading import revert_weight_conversion
    except ModuleNotFoundError:
        return state_dict

    return revert_weight_conversion(model, state_dict)
