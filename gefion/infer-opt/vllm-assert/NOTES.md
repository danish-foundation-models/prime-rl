# vLLM 0.28 prefill crash: `assert num_new_tokens > 0` (scheduler.py:967)

Crashes: job 534519 attempt_4 node_1 (dgx046) at 15:33:26; job 534523 attempt_5 node_2 at 15:50:17. Same traceback in both.

## Root cause
Order of operations for a waiting request in `Scheduler.schedule()` on a **prefill** engine (NIXL, hybrid Mamba, prefix caching):
1. `_get_local_prefix_cache_hit(request)` caps the local hit at `num_tokens - 1 = N - 1`. The hit is a multiple of the 1040-token block. hash_block_size = gcd(group sizes) = 1040, so there are no partial hits.
2. `connector.get_num_new_matched_tokens()` -> `NixlPullScheduler` -> `_truncate_mamba_request_for_prefill()`
   (`nixl/base_scheduler.py:379`) pops the last prompt token for `do_remote_decode` requests, so now `num_tokens = N - 1`.
3. If the whole N-1 prefix is cached, which needs N-1 to be a multiple of 1040 (N ≡ 1 mod 1040), then
   `num_computed_tokens = N-1 = num_tokens`. The `<=` assert passes, `num_new_tokens = 0`, and the assert at line 967 fails.

When the full N-1 prefix is cached: an exact duplicate prompt reaches the same prefill engine after the
first copy was prefilled. The first copy computes exactly N-1 tokens, which is block-aligned, so the Mamba state is
checkpointed at N-1. Sources of duplicates: group-of-8 siblings with identical trajectories, which are more common at
T=0.8, and harness retries. Turns that extend a prompt can't trigger it, because only N-1 of the previous prompt is cached.

Upstream: fixed by vllm#53523 (merged 2026-08-25; it moves the P-side truncation into `on_new_request`, before the lookup;
issues #53514, #53706). The installed v0.28.0 wheel (from the PyPI release) does not contain it. vllm#54411 is still open. It
would turn the assert into FINISHED_ERROR for the request.

## Patch: `scheduler-p-side-truncation-full-hit.diff`
Apply with: `cd .venv/lib/python3.12/site-packages && patch -p1 < <this dir>/scheduler-p-side-truncation-full-hit.diff`
(a dry run against the venv applies cleanly; the patched file compiles).

After the connector call, if there are no external tokens and the local hit is >= the (possibly truncated)
`num_tokens`, redo `kv_cache_manager.get_computed_blocks(request)`. That lookup caps at the new `num_tokens - 1`,
so the last 1040-token block is recomputed (k-1 blocks hit, 1040 new tokens, ending block-aligned at N-1 so that h(N-1) is
produced for D). This mirrors the existing `hit_diverged` reconcile two lines above. Lookups are side-effect free:
no refcount or touch until `allocate_slots`. The patch is a no-op unless the crash condition holds.
Cost: one extra block of prefill compute for each affected duplicate (about 1/1040 of duplicates).

## Not causes
The relay storm and cleanup (attempt_5 crashed with no relay event), pause/resume (`mode=keep`), DP prefill throttle
(DP=1), spec decode (none), and the encoder path (text only).

## Separate observation
`/pause?mode=keep&clear_cache=false`: the prefix cache, including Mamba states, survives weight updates, so prefill reuses KV
and Mamba states computed with the old policy weights. The prefix hit rate stays at about 86% across updates.
