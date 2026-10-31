"""Make vLLM 0.28's native CPU offload (OffloadingConnector) keep multi-turn history for Nemotron-H.

Two gaps, measured with bench/cache_probe.py (job 531395, FP8, Mamba "align", 96 GiB/engine host tier):

1. Decoded tokens are never offloaded. OffloadingSpec reads ``offload_prompt_only`` from
   kv_connector_extra_config with default True ("useful when prior turns' generated tokens are
   dropped before the next turn"). PRIME-RL's native config does not set it, so a re-sent turn whose
   history is mostly the previous completion gets 0 host hits, with or without MTP. The GPU prefix
   cache is unaffected (it caches decoded blocks either way).
   Fix: default it to False (INFER_OPT_OFFLOAD_PROMPT_ONLY=1 restores the stock default).

2. With MTP, every KV cache group is treated as a drafter group. vLLM only annotates the draft
   layer's group for DeepSeek-V4 (_annotate_eagle_groups_deepseek_v4); otherwise the coordinator
   and the offload scheduler fall back to flagging all groups, so the Mamba groups also drop their
   "volatile" trailing block. On the host tier that cost one extra 4176-token chunk per hit (8352
   vs 12528 GPU-cached tokens of a 20K prompt) and, while decoding, holds each Mamba chunk back
   until the next one completes. Upstream removed the offload-side fallback in
   vllm-project/vllm#52771 (merged after 0.28/0.29).
   Fix: flag only the group(s) holding the MTP draft layers (layer names containing "mtp"), as the
   DeepSeek-V4 path does. The drafter's own KV keeps the drop; the target model's groups do not
   need it (the drop exists because the EAGLE/MTP lookahead token pollutes the draft layer's KV).

3. Backports into the offload scheduler (offloading/scheduler.py, applied as exact source edits):
   - vllm-project/vllm#52771 (merged): lift the drafter group's trailing-chunk exclusion once the
     request has finished, and widen the lookup by one chunk for every drafter group, not only
     sliding-window ones (without it the drafter's pop can shrink the hit below a coarser sibling
     group's chunk and zero the whole request's hit: our decoded-history probe got 0 tokens).
   - vllm-project/vllm#54288 (merged): at finish, store only up to the GPU prefix cache's commit
     point (num_tokens - 1); the last sampled token's slot can hold a rejected draft's KV.
   - vllm-project/vllm#52807 (merged): on load, scan for the first fresh block from the computed
     boundary, not 0 (Mamba groups hold non-null unhashed blocks below it). Its new
     ``num_locally_computed_tokens % tokens_per_block == 0`` assert is not taken: with
     prefix_match_unit < block size, partial local hits are legitimate.
   INFER_OPT_OFFLOAD_BACKPORTS=0 disables them.

Enable with INFER_OPT=offload_fix.
"""
import os

from infer_opt import _log, patch_source, when_imported

_SCHED = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler"
_BACKPORTS = [
    # 52771: no all-groups drafter fallback (only annotated groups are drafter groups)
    ("""        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))
""",
     """        # infer_opt backport of vllm#52771: no all-groups drafter fallback
"""),
    # 52771: finished requests have no pending speculation
    ("if group_config.is_eagle_group and is_decoding:",
     "if group_config.is_eagle_group and is_decoding and not self.req.is_finished():"),
    # 52771: widen the query for every drafter group
    ("if is_eagle_unverified and sliding_window_size_in_chunks is not None:",
     "if is_eagle_unverified:"),
    # 54288: clamp finish-time stores to the prefix cache's commit point
    ("                num_tokens_after_batch = req.num_tokens\n",
     "                num_tokens_after_batch = max(req.num_prompt_tokens, req.num_tokens - 1)\n"),
    # 52807: load region starts at the first fresh block at/after the computed boundary
    ("""            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break
""",
     """            num_locally_computed_gpu_blocks = num_gpu_blocks
            # infer_opt backport of vllm#52807: scan from the computed boundary, not 0 (sparse
            # Mamba groups legitimately hold non-null unhashed blocks below it).
            for i in range(cdiv(num_locally_computed_tokens, tokens_per_block), num_gpu_blocks):
                block = group_blocks[i]
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break
"""),
]


def _flag_mtp_groups(vllm_config, kv_cache_groups):
    if os.environ.get("INFER_OPT_MTP_KEEP_LAST_BLOCK") == "1":
        return  # no drafter groups at all (infer_opt.mtp_keep_last_block)
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None or not spec.use_eagle() or any(g.is_eagle_group for g in kv_cache_groups):
        return
    hits = [i for i, g in enumerate(kv_cache_groups) if any("mtp" in n for n in g.layer_names)]
    if not hits:
        _log("offload_fix: no KV cache group holds an 'mtp' layer; keeping vLLM's all-groups fallback")
        return
    for i in hits:
        kv_cache_groups[i].is_eagle_group = True
    _log(f"offload_fix: drafter KV groups {hits} of {len(kv_cache_groups)} "
         f"(e.g. {[n for n in kv_cache_groups[hits[0]].layer_names if 'mtp' in n][:2]})")


def install() -> None:
    prompt_only = os.environ.get("INFER_OPT_OFFLOAD_PROMPT_ONLY", "0") == "1"

    def patch_utils(mod):
        orig = mod.get_kv_cache_groups

        def get_kv_cache_groups(vllm_config, kv_cache_spec, *a, **kw):
            groups = orig(vllm_config, kv_cache_spec, *a, **kw)
            _flag_mtp_groups(vllm_config, groups)
            return groups

        mod.get_kv_cache_groups = get_kv_cache_groups

        def patch_runner(runner_mod):
            if getattr(runner_mod, "get_kv_cache_groups", None) is orig:
                runner_mod.get_kv_cache_groups = get_kv_cache_groups

        when_imported("vllm.v1.worker.gpu_model_runner", patch_runner)

    def patch_spec(mod):
        S = mod.OffloadingSpec
        orig_init = S.__init__

        def __init__(self, config, *a, **kw):
            orig_init(self, config, *a, **kw)
            if "offload_prompt_only" not in self.extra_config:
                self.offload_prompt_only = prompt_only

        S.__init__ = __init__

    if os.environ.get("INFER_OPT_OFFLOAD_BACKPORTS", "1") == "1":
        patch_source(_SCHED, _BACKPORTS, "offload_fix")
    when_imported("vllm.v1.core.kv_cache_utils", patch_utils)
    when_imported("vllm.v1.kv_offload.base", patch_spec)
    _log(f"offload_fix: offload_prompt_only={prompt_only}, MTP drafter-group annotation on")
