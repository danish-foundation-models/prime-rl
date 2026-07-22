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

- an organization-owned fork with an `upstream` remote;
- a maintained `feat/lumi-integration` branch based on current upstream;
- a focused commit series without experiments or generated files;
- an exact parent submodule pin;
- a short entry in this document describing the behavior, validation command,
  and condition for removing the fork.

Do not commit dependency work from a detached submodule checkout. Refresh and
classify it in its own repository first, then update the PRIME-RL pin as a
separate commit. The current integration uses upstream's pins for `renderers`,
`research-environments`, and `verifiers`; their existing local work is not part
of this branch.
