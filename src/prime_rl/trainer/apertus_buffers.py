"""Identify Apertus buffers that the native sharded loader can restore."""


def can_reinit_apertus_buffers(model):
    if getattr(getattr(model, "config", None), "model_type", None) != "apertus":
        return False
    buffers = {name for name, _ in model.named_buffers()}
    # xIELU beta/eps are persistent checkpoint entries, not missing constants.
    # DCP must load them; unknown nonpersistent buffers still force the fallback.
    persistent = set(model.state_dict())
    rope = {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}
    return "model.rotary_emb.inv_freq" in buffers and buffers - persistent <= rope


def normalize_apertus_parameter_dtype(model, dtype):
    """Honor optimizer dtype for xIELU parameters with explicit config dtypes."""
    if getattr(getattr(model, "config", None), "model_type", None) != "apertus":
        return
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)
