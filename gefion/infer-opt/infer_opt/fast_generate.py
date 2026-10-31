"""Cheaper /inference/v1/generate responses with many top-logprobs (PRIME-RL score centering).

PRIME-RL asks for ``logprobs = top_k + 1 = 33`` per generated token. In stock vLLM 0.28 the
single API-server process then, for every token of every request,
  - detokenizes all 33-34 candidate tokens (LogprobsProcessor) although the tokens-in/tokens-out
    endpoint only ever emits ``"token_id:<id>"`` strings,
  - keeps a dict of ``Logprob`` dataclasses per position, and at the end
  - builds ~34 pydantic models per token, ``model_dump()``s them and ``json.dumps`` the result
    (a 16K-token answer is ~550K small objects and ~25 MB of JSON).
This patch, for generate requests that ask for logprobs and have no stop *strings*:
  - sets ``detokenize=False`` (no text is returned by this endpoint; stop token ids still work),
  - stores the per-step logprob rows in one growing numpy buffer per request (``NpLogprobs``)
    instead of Python floats/ints: with 768 running requests x thousands of tokens x 34 entries
    the stock containers hold 10^8 Python objects, and full GC passes over them stall the API
    server's event loop for tens of seconds (seen as 30-60 s gaps between 10 s stats lines),
  - serialises the response directly from those buffers with orjson.
The JSON has the same keys and values as stock (same top-logprob order and dedup semantics,
same -9999 clamp), so renderers/client.py parses it unchanged.

Enable with INFER_OPT=fast_generate (API-server side only).
"""
import asyncio

from infer_opt import _log, when_imported

CLAMP = -9999.0


class _Raw(dict):
    """model_dump() result that carries pre-serialised JSON bytes."""

    def __init__(self, raw: bytes):
        super().__init__()
        self.raw = raw


def _np_logprobs_cls():
    import numpy as np
    from vllm.logprobs import FlatLogprobs, Logprob

    class NpLogprobs(FlatLogprobs):
        """Sample logprobs as [positions, k+1] numpy buffers (column 0 = sampled token).

        Behaves like FlatLogprobs for any other reader (len / index / iterate / slice), but the
        hot path never creates per-entry Python objects."""

        def __init__(self):
            super().__init__()
            self.n, self.tok, self.lp = 0, None, None

        def add(self, ids, lps):
            m, w = ids.shape
            if self.tok is None:
                cap = max(256, m)
                self.tok = np.empty((cap, w), np.int32)
                self.lp = np.empty((cap, w), np.float32)
            elif self.n + m > self.tok.shape[0]:
                cap = max(self.n + m, 2 * self.tok.shape[0])
                self.tok = np.resize(self.tok, (cap, self.tok.shape[1]))
                self.lp = np.resize(self.lp, (cap, self.lp.shape[1]))
            self.tok[self.n:self.n + m] = ids
            self.lp[self.n:self.n + m] = lps
            self.n += m

        def __len__(self):
            return self.n

        def _pos(self, i):
            return {int(t): Logprob(logprob=float(v)) for t, v in zip(self.tok[i], self.lp[i])}

        def __getitem__(self, index):
            if isinstance(index, int):
                return self._pos(index if index >= 0 else self.n + index)
            out = FlatLogprobs()
            for i in range(*index.indices(self.n)):
                out.append(self._pos(i))
            return out

        def __iter__(self):
            return (self._pos(i) for i in range(self.n))

        def append(self, logprobs_one_position):  # not used on the patched path
            raise NotImplementedError("NpLogprobs is filled by add()")

    return NpLogprobs


_NP = None


def _content(token_ids, lps, num: int) -> list:
    """Stock _create_tokens_logprobs semantics, as plain dicts, from FlatLogprobs or list[dict]."""
    from vllm.logprobs import FlatLogprobs

    keep = max(num, 1)
    out = []
    append = out.append
    if _NP is not None and isinstance(lps, _NP):
        T, L = lps.tok[:lps.n].tolist(), lps.lp[:lps.n].tolist()
        for i, tid in enumerate(token_ids):
            pos = dict(zip(T[i], L[i]))  # first-seen order, last value: same as the stock dict build
            v = pos.get(tid)
            if v is None:
                append({"token": f"token_id:{tid}", "logprob": CLAMP, "bytes": None, "top_logprobs": []})
                continue
            top = [{"token": f"token_id:{t}", "logprob": x if x > CLAMP else CLAMP, "bytes": None}
                   for t, x in list(pos.items())[:keep]]
            append({"token": f"token_id:{tid}", "logprob": v if v > CLAMP else CLAMP, "bytes": None, "top_logprobs": top})
    elif isinstance(lps, FlatLogprobs):
        S, E, T, L = lps.start_indices, lps.end_indices, lps.token_ids, lps.logprobs
        for i, tid in enumerate(token_ids):
            pos = {}
            for j in range(S[i], E[i]):  # dict: first-seen order, last value (== stock dict build)
                pos[T[j]] = L[j]
            v = pos.get(tid)
            if v is None:
                append({"token": f"token_id:{tid}", "logprob": CLAMP, "bytes": None, "top_logprobs": []})
                continue
            top = [{"token": f"token_id:{t}", "logprob": x if x > CLAMP else CLAMP, "bytes": None}
                   for t, x in list(pos.items())[:keep]]
            append({"token": f"token_id:{tid}", "logprob": v if v > CLAMP else CLAMP, "bytes": None, "top_logprobs": top})
    else:
        for i, tid in enumerate(token_ids):
            pos = lps[i]
            if pos is None or pos.get(tid) is None:
                append({"token": f"token_id:{tid}", "logprob": CLAMP, "bytes": None, "top_logprobs": []})
                continue
            v = pos[tid].logprob
            top = [{"token": f"token_id:{t}", "logprob": max(lp.logprob, CLAMP), "bytes": None}
                   for t, lp in list(pos.items())[:keep]]
            append({"token": f"token_id:{tid}", "logprob": max(v, CLAMP), "bytes": None, "top_logprobs": top})
    return out


def install() -> None:
    def patch_serving(mod):
        import orjson
        from vllm.entrypoints.openai.engine.protocol import PromptTokenUsageInfo, UsageInfo
        from vllm.entrypoints.scale_out.token_in_token_out.protocol import (GenerateResponse,
                                                                            GenerateResponseChoice)

        S = mod.ServingTokens
        orig_serve, orig_full = S.serve_tokens, S.serve_tokens_full_generator

        class FastGenerateResponse(GenerateResponse):
            def model_dump(self, *a, **k):  # the route calls model_dump() and wraps it in JSONResponse
                return _Raw(orjson.dumps(self.__pydantic_private__["_payload"]))

        async def serve_tokens(self, request, raw_request=None):
            sp = request.sampling_params
            if sp is not None and sp.logprobs is not None and not sp.stop and not request.stream:
                sp.detokenize = False
                sp.flat_logprobs = True
            return await orig_serve(self, request, raw_request)

        async def serve_tokens_full_generator(self, request, result_generator, request_id, model_name,
                                              request_metadata):
            sp = request.sampling_params
            if (sp.logprobs is None or self.model_config.enable_return_routed_experts
                    or (self.enable_log_outputs and self.request_logger)):
                return await orig_full(self, request, result_generator, request_id, model_name, request_metadata)
            import time

            created = int(time.time())
            final = None
            try:
                async for res in result_generator:
                    final = res
            except asyncio.CancelledError:
                return self.create_error_response("Client disconnected")
            assert final is not None
            choices, contents, n_gen = [], [], 0
            for o in final.outputs:
                contents.append(_content(o.token_ids, o.logprobs, sp.logprobs))
                choices.append(GenerateResponseChoice(
                    index=o.index, logprobs=None, finish_reason=o.finish_reason if o.finish_reason else "stop",
                    token_ids=list(o.token_ids), routed_experts=None,
                    sampling_mask=o.sampling_mask.token_ids if o.sampling_mask is not None else None))
                n_gen += len(o.token_ids)
            n_prompt = len(final.prompt_token_ids) + len(final.encoder_prompt_token_ids or [])
            usage = UsageInfo(prompt_tokens=n_prompt, completion_tokens=n_gen, total_tokens=n_prompt + n_gen)
            if self.enable_prompt_tokens_details and final.num_cached_tokens is not None:
                usage.prompt_tokens_details = PromptTokenUsageInfo(cached_tokens=final.num_cached_tokens)
            request_metadata.final_usage_info = usage
            from vllm.entrypoints.scale_out.token_in_token_out.serving import clamp_prompt_logprobs

            resp = GenerateResponse(request_id=request_id, created=created, model=model_name, choices=choices,
                                    usage=usage, prompt_logprobs=clamp_prompt_logprobs(final.prompt_logprobs),
                                    kv_transfer_params=final.kv_transfer_params,
                                    ec_transfer_params=final.ec_transfer_params)
            payload = resp.model_dump()
            for c, lp in zip(payload["choices"], contents):
                c["logprobs"] = {"content": lp}
            fast = FastGenerateResponse.model_construct(**dict(resp))
            object.__setattr__(fast, "__pydantic_private__", {"_payload": payload})
            return fast

        S.serve_tokens = serve_tokens
        S.serve_tokens_full_generator = serve_tokens_full_generator
        _log("fast_generate: ServingTokens patched")

    def patch_router(mod):
        from starlette.responses import Response

        orig = mod.JSONResponse

        def JSONResponse(content=None, *a, **k):
            if isinstance(content, _Raw):
                return Response(content.raw, *a, media_type="application/json", **k)
            return orig(content, *a, **k)

        mod.JSONResponse = JSONResponse

    def patch_logprobs(mod):
        global _NP
        _NP = _np_logprobs_cls()
        P = mod.LogprobsProcessor
        orig_new, orig_update = P.from_new_request.__func__, P._update_sample_logprobs

        @classmethod
        def from_new_request(cls, tokenizer, request):
            self = orig_new(cls, tokenizer, request)
            sp = request.sampling_params
            if (tokenizer is None and sp is not None and sp.flat_logprobs and self.num_logprobs is not None
                    and self.num_logprobs >= 0):
                self.logprobs = _NP()
            return self

        def _update_sample_logprobs(self, logprobs_lists):
            if not isinstance(self.logprobs, _NP):
                return orig_update(self, logprobs_lists)
            ids, lps = logprobs_lists[0], logprobs_lists[1]
            w = self.num_logprobs + 1
            self.logprobs.add(ids[:, :w], lps[:, :w])
            self.cumulative_logprob += float(lps[:, 0].sum())

        P.from_new_request = from_new_request
        P._update_sample_logprobs = _update_sample_logprobs
        _log("fast_generate: LogprobsProcessor uses numpy buffers for flat logprobs")

    when_imported("vllm.v1.engine.logprobs", patch_logprobs)
    when_imported("vllm.entrypoints.scale_out.token_in_token_out.serving", patch_serving)
    when_imported("vllm.entrypoints.scale_out.token_in_token_out.api_router", patch_router)
