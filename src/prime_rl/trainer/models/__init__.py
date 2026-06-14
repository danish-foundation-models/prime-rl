## a bit of context here, this basically copy AutoModelForCausalLM from transformers, but use our own model instead

from collections import OrderedDict

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous
from prime_rl.trainer.models.llama import LlamaForCausalLM
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM
try:
    from prime_rl.trainer.models.afmoe import AfmoeConfig, AfmoeForCausalLM
except (ImportError, OSError, AttributeError):
    AfmoeConfig = None
    AfmoeForCausalLM = None
try:
    from prime_rl.trainer.models.glm4_moe import Glm4MoeConfig, Glm4MoeForCausalLM
except (ImportError, OSError, AttributeError):
    Glm4MoeConfig = None
    Glm4MoeForCausalLM = None
try:
    from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
except (ImportError, OSError, AttributeError):
    GlmMoeDsaConfig = None
    GlmMoeDsaForCausalLM = None
try:
    from prime_rl.trainer.models.gpt_oss import GptOssConfig, GptOssForCausalLM
except (ImportError, OSError, AttributeError):
    GptOssConfig = None
    GptOssForCausalLM = None
try:
    from prime_rl.trainer.models.laguna import LagunaConfig, LagunaForCausalLM
except (ImportError, OSError, AttributeError):
    LagunaConfig = None
    LagunaForCausalLM = None
try:
    from prime_rl.trainer.models.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM
except (ImportError, OSError, AttributeError):
    MiniMaxM2Config = None
    MiniMaxM2ForCausalLM = None
try:
    from prime_rl.trainer.models.nemotron_h import NemotronHConfig, NemotronHForCausalLM
except (ImportError, OSError):
    NemotronHConfig = None
    NemotronHForCausalLM = None
try:
    from prime_rl.trainer.models.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM
except (ImportError, OSError):
    Qwen3_5MoeConfig = None
    Qwen3_5MoeForCausalLM = None
try:
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
except (ImportError, OSError):
    Qwen3MoeConfig = None
    Qwen3MoeForCausalLM = None

# Make custom config discoverable by AutoConfig
if AfmoeConfig is not None:
    AutoConfig.register("afmoe", AfmoeConfig, exist_ok=True)
if Glm4MoeConfig is not None:
    AutoConfig.register("glm4_moe", Glm4MoeConfig, exist_ok=True)
if GlmMoeDsaConfig is not None:
    AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig, exist_ok=True)
if LagunaConfig is not None:
    AutoConfig.register("laguna", LagunaConfig, exist_ok=True)
if MiniMaxM2Config is not None:
    AutoConfig.register("minimax_m2", MiniMaxM2Config, exist_ok=True)
if NemotronHConfig is not None:
    AutoConfig.register("nemotron_h", NemotronHConfig, exist_ok=True)
if Qwen3MoeConfig is not None:
    AutoConfig.register("qwen3_moe", Qwen3MoeConfig, exist_ok=True)
if Qwen3_5MoeConfig is not None:
    AutoConfig.register("qwen3_5_moe_text", Qwen3_5MoeConfig, exist_ok=True)
# GptOssConfig is just HF's class - already registered by transformers, no override needed.

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
_CUSTOM_CAUSAL_LM_MAPPING.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)
_CUSTOM_CAUSAL_LM_MAPPING.register(Qwen3Config, Qwen3ForCausalLM, exist_ok=True)
if AfmoeConfig is not None and AfmoeForCausalLM is not None:
    _CUSTOM_CAUSAL_LM_MAPPING.register(AfmoeConfig, AfmoeForCausalLM, exist_ok=True)
if Glm4MoeConfig is not None and Glm4MoeForCausalLM is not None:
    _CUSTOM_CAUSAL_LM_MAPPING.register(Glm4MoeConfig, Glm4MoeForCausalLM, exist_ok=True)
if GlmMoeDsaConfig is not None and GlmMoeDsaForCausalLM is not None:
    _CUSTOM_CAUSAL_LM_MAPPING.register(GlmMoeDsaConfig, GlmMoeDsaForCausalLM, exist_ok=True)
if LagunaConfig is not None and LagunaForCausalLM is not None:
    _CUSTOM_CAUSAL_LM_MAPPING.register(LagunaConfig, LagunaForCausalLM, exist_ok=True)
if MiniMaxM2Config is not None and MiniMaxM2ForCausalLM is not None:
    _CUSTOM_CAUSAL_LM_MAPPING.register(MiniMaxM2Config, MiniMaxM2ForCausalLM, exist_ok=True)
if NemotronHConfig is not None and NemotronHForCausalLM is not None:
    _CUSTOM_CAUSAL_LM_MAPPING.register(NemotronHConfig, NemotronHForCausalLM, exist_ok=True)
if Qwen3MoeConfig is not None and Qwen3MoeForCausalLM is not None:
    _CUSTOM_CAUSAL_LM_MAPPING.register(Qwen3MoeConfig, Qwen3MoeForCausalLM, exist_ok=True)
if Qwen3_5MoeConfig is not None and Qwen3_5MoeForCausalLM is not None:
    _CUSTOM_CAUSAL_LM_MAPPING.register(Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM, exist_ok=True)
if GptOssConfig is not None and GptOssForCausalLM is not None:
    _CUSTOM_CAUSAL_LM_MAPPING.register(GptOssConfig, GptOssForCausalLM, exist_ok=True)


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def supports_custom_impl(model_config: PretrainedConfig) -> bool:
    """Check if the model configuration supports the custom PrimeRL implementation.

    Args:
        model_config: The model configuration to check.

    Returns:
        True if the model supports custom implementation, False otherwise.
    """
    return type(model_config) in _CUSTOM_CAUSAL_LM_MAPPING


# Mapping from HF composite VLM model_type to custom PrimeRL class.
# Used by get_model() to dispatch VLMs that have a custom text model implementation.
# Points to the same unified class — the config drives text-only vs VLM behavior.
_CUSTOM_VLM_MAPPING: dict[str, type] = {}
if Qwen3_5MoeForCausalLM is not None:
    _CUSTOM_VLM_MAPPING["qwen3_5_moe"] = Qwen3_5MoeForCausalLM


def get_custom_vlm_cls(model_config: PretrainedConfig) -> type | None:
    """Return the custom PrimeRL VLM class for this config, or None if unsupported."""
    return _CUSTOM_VLM_MAPPING.get(getattr(model_config, "model_type", None))


__all__ = [
    "AutoModelForCausalLMPrimeRL",
    "PreTrainedModelPrimeRL",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]
