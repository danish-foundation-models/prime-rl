"""Keep the last cached block on prefix hits with MTP (no EAGLE "last-block drop").

With an EAGLE/MTP drafter, vLLM 0.28 drops the last matching block of every prefix hit and, in Mamba
"align" mode, never checkpoints the state at a prefill's last block boundary
(Scheduler._mamba_block_aligned_split backs off one block when use_eagle). The reason: the drafter's
KV at the hit's last position was written with the *next* token of the request that produced it, which
the new request may not share. With 4176-token blocks that costs a whole block per hit: a re-sent turn
only reuses anything past 8352 tokens of context, so on the multi-turn load FP8 + MTP got 0.3-0.5%
prefix hits (jobs 531400, 531424) against 36% without MTP.

That stale entry only feeds the drafter. The target model's KV and Mamba state are its own, and the
rejection sampler verifies every draft against the target, so outputs and logprobs are unchanged; at
worst the first draft after a hit is accepted less often. (vllm-project/vllm#52771 removes the offload
connector's all-groups drafter fallback on the same reasoning.) For a multi-turn continuation the next
token is the same anyway.

Source edits (exact; any mismatch leaves that module stock):
  - kv_cache_coordinator: no drafter groups, so no group gets the last-block drop.
  - sched/scheduler: no one-block back-off of the last cacheable Mamba checkpoint.
Also tells offload_fix not to flag the MTP group, so the host tier matches the GPU cache.

Enable with INFER_OPT=mtp_keep_last_block (list it before offload_fix).
"""
import os

from infer_opt import patch_source

_COORD = [(
    """        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
""",
    """        # infer_opt.mtp_keep_last_block: no EAGLE/MTP last-block drop for any group.
        self.eagle_group_ids = set()
""")]

_SCHED = [(
    """        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
""",
    """        # infer_opt.mtp_keep_last_block: no EAGLE back-off (the drop it compensates for is off)
""")]


def install() -> None:
    os.environ["INFER_OPT_MTP_KEEP_LAST_BLOCK"] = "1"  # read by offload_fix
    patch_source("vllm.v1.core.kv_cache_coordinator", _COORD, "mtp_keep_last_block")
    patch_source("vllm.v1.core.sched.scheduler", _SCHED, "mtp_keep_last_block")
