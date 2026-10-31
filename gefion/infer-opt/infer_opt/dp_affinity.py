"""Prefix-affinity routing across vLLM's internal DP engines (DP8 in one server).

Stock DPLBAsyncMPClient sends every request to the least-loaded engine. Prefix caching and the host
KV offload tier are per engine, so the next turn of a multi-turn rollout (and the other samples of a
GRPO group) lands on a random engine and finds its prefix cached only ~1/8 of the time.

This patch routes by a hash of the first INFER_OPT_DP_AFFINITY_TOKENS prompt tokens (default 256;
a shorter prompt is hashed whole, so set it at or below the shortest first-turn prompt). All turns
of a rollout share those tokens, so they go to the same engine. Load stays bounded: the affinity
engine is taken only while its load (waiting + running, as the stock scorer sees it) is at most
INFER_OPT_DP_AFFINITY_SLACK (default 1.25) x the mean plus 8; otherwise the stock choice is used.
An explicit data_parallel_rank (X-data-parallel-rank header) always wins.

Enable with INFER_OPT=dp_affinity.
"""
import hashlib
import os

from infer_opt import _log, when_imported


def install() -> None:
    k = int(os.environ.get("INFER_OPT_DP_AFFINITY_TOKENS", "256"))
    slack = float(os.environ.get("INFER_OPT_DP_AFFINITY_SLACK", "1.25"))

    def patch(mod):
        C = mod.DPLBAsyncMPClient
        orig = C.get_core_engine_for_request
        stats = {"affinity": 0, "fallback": 0}

        def get_core_engine_for_request(self, request):
            ids = request.prompt_token_ids
            if request.data_parallel_rank is not None or not ids:
                return orig(self, request)
            n = len(self.core_engines)
            head = ids[:k]
            digest = hashlib.blake2b(b"".join(int(t).to_bytes(4, "little") for t in head), digest_size=8).digest()
            idx = int.from_bytes(digest, "little") % n
            loads = [max(self.client_count * self.engine_inflight[e], c[0] + c[1])
                     for e, c in zip(self.core_engines, self.lb_engines)]
            if loads[idx] > slack * sum(loads) / n + 8:
                stats["fallback"] += 1
                return orig(self, request)
            stats["affinity"] += 1
            if (stats["affinity"] + stats["fallback"]) % 5000 == 0:
                _log(f"dp_affinity: {stats}")
            self.lb_engines[idx][0] += self.client_count
            chosen = self.core_engines[idx]
            self.reqs_in_flight[request.request_id] = chosen
            self.engine_inflight[chosen] += 1
            return chosen

        C.get_core_engine_for_request = get_core_engine_for_request
        _log(f"dp_affinity: routing by the first {k} prompt tokens, slack {slack}")

    when_imported("vllm.v1.engine.core_client", patch)
