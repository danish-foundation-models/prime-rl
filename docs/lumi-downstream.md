# LUMI downstream maintenance

The LUMI integration is a deliberately small downstream layer on top of
PRIME-RL. It contains cluster runtime and orchestration behavior that is not
expected to be accepted upstream. Model run specifications, generated outputs,
debug probes, and historical compatibility patches do not belong on the
integration branch.

## Branch policy

- `upstream/main` is the canonical PRIME-RL source.
- `feat/lumi-integration` is the maintained LUMI integration branch.
- Bring upstream changes into the integration branch with a merge. Do not
  rewrite a published integration branch while jobs or collaborators may refer
  to its commits.
- Keep downstream changes in small commits grouped by runtime, orchestration,
  or a single compatibility requirement.
- Keep experiment configs and run records on `exp/*` branches or in the `post`
  repository.

The integration branch currently carries:

- immutable-image LUMI submit wrappers and Slurm templates;
- project-local cache, metadata, and artifact placement;
- ROCm GPU visibility and NVML-tolerant launch behavior;
- the ROCm-safe fp32 lm-head inference path;
- serialized Hugging Face snapshot downloads for the shared model cache;
- a small regular-worker router because the operational image does not provide
  `vllm-router`;
- UCloud image preparation, capacity prewarming, and cleanup orchestration.
- an opt-in, revision-pinned source overlay for the gfx90a vLLM stack, preserving
  the runtime image's compiled vLLM extensions while replacing its tuned Python
  modules.

The integration branch does not carry the old writable-overlay builder,
historical smoke/run configs, gradient diagnostics, packed-SDPA support, or
speculative model compatibility patches. LUMI configs use FlashAttention.

## Refresh procedure

1. Check active jobs and preserve the current working tree.
2. Fetch `upstream/main` and merge it into `feat/lumi-integration`.
3. Review the resulting diff against upstream. A refresh must not reintroduce
   experiment configs, generated state, or retired runtime paths.
4. Run `git diff --check` and `bash -n` for changed shell launchers.
5. Dry-run the generic SFT and RL configs and inspect their rendered Slurm
   scripts.
6. Run the focused unit tests for changed Python components.
7. Record any new dependency fork pin under the dependency policy below.

## Dependency policy

Use the submodule commits selected by upstream unless a tested LUMI requirement
cannot be implemented in PRIME-RL itself. A downstream dependency requires all
of the following:

- a maintainer-owned fork under `rlrs` with an `upstream` remote;
- a maintained `feat/lumi-integration` branch based on current upstream;
- a focused commit series without experiments or generated files;
- an exact parent submodule pin;
- a short entry in this document describing the behavior, validation command,
  and condition for removing the fork.

Do not commit dependency work from a detached submodule checkout. Refresh and
classify it in its own repository first, then update the PRIME-RL pin as a
separate commit.

Current downstream dependency pins:

- `renderers` `407d403`: lets the default renderer skip a system-only
  attribution prefix while continuing to reject malformed prefixes after the
  first user message. Validated by its two focused default-renderer tests.
  Remove when upstream provides the same guarded behavior.
- `verifiers` `ab3adefa4`: adds the UCloud sandbox runtime and polling-relay
  interception backend, uses the current typed sandbox SDK request, supports
  gateway-managed image IDs, and defaults to the production relay URL.
  Validated by the focused sandbox/relay tests, lint, syntax compilation, and
  a live managed-image create/exec/delete smoke. Remove when verifiers supports
  a pluggable UCloud provider or equivalent upstream runtime and relay.
- `ucloud-sandboxes-sdk` `96409abd`: supplies the public registration-token
  relay API used by the Verifiers adapter. It is pinned as live source because
  the operational SIF does not bundle the SDK. Remove the submodule only after
  the supported runtime image provides the same API.
- `research-environments` stays on the upstream pin. The preserved
  `mini-swe-agent-v2` work is being separated into its own package rather than
  maintained as a fork of a repository that removed it.

## gfx90a vLLM source overlay

Set `PRIME_RL_VLLM_SOURCE_ROOT` and `PRIME_RL_VLLM_SOURCE_REV` in a LUMI RL
config's `[env_vars]` table to use a committed vLLM tuning stack. The launcher
binds only the stack's Python and tuning files over vLLM 0.24.0, so the SIF's
compiled extensions remain available. Set `PRIME_RL_AITER_SITE_PACKAGES` when
the stack also needs an external AITER layer. The launcher fails before `srun`
if the checkout, revision, source files, or AITER layer do not match.

The UCloud relay URL defaults to
`https://app-sandboxes-relay-v2.cloud.sdu.dk`. Relay deployments intentionally
use two distinct opaque bearer tokens: `UCLOUD_RELAY_SANDBOX_TOKEN` for model
calls made inside sandboxes and `UCLOUD_RELAY_WORKER_TOKEN` for trusted rollout
registration and polling. These are deployment tokens, not JWT credentials.
