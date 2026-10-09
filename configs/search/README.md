# Search environments

OpenSeeker's TinyFish configuration is a four-task provider trial using the final
Super checkpoint and the plain null harness on a local subprocess runtime. It has
not been activated in the RL mixture. Free TinyFish quota is insufficient for full
rollout concurrency; infrastructure failure handling must also pass end-to-end
validation before training. Its shared cache/usage ledger lives in `.cache/tinyfish`.

BrowseComp-Plus uses its held-out 830-task test split, a shared local BM25 index,
and a restricted UCloud runtime for the null harness. The task's internet denial
requires a network-policy-capable runtime even for this harness; subprocess is
rejected by the framework. Core policy enforcement is unchanged. No sandbox or
model evaluation is started by dataset preparation. The 32-task config is an online
smoke cohort, not a full benchmark score. GLM 5.3 Flash grading and BM25s retrieval
must be reported when comparing with published scores.

Run with the search dependency layer, `BROWSECOMP_PLUS_CACHE_DIR` set to
`/scratch/project_465002751/rasmus/.cache/browsecomp-plus`, and the bench runtime.
The layer is `/scratch/project_465002751/rasmus/.runtime/layers/search-agentic-20260921`.
Dataset revisions and fingerprints are recorded in
`artifacts/training/nemotron-super-r5-rl-scale-20260917/search-20260921/data-manifest.json`.

Before running either config, provide an inference endpoint, load judge credentials
server-side, and validate the native EvalConfig. BrowseComp-Plus also needs working
UCloud relay transport. Do not use its test questions for training.

## Monid trial

`openseeker_monid.toml` uses the final Super checkpoint with Monid's free TinyFish
search/fetch endpoints. On 2026-09-21, 32 live search requests completed in 19.52s
at up to eight concurrent calls (median 2.70s, max 3.67s, zero failures). This is
a bounded burst check, not proof of an unlimited/hourly quota. Raw responses,
inspection metadata, and timing are in `search-20260921/monid-*.json` alongside
the dataset manifest. The native tools also passed shared-request coalescing and
page-fetch checks. OpenSeeker's task policy explicitly permits internet access.

The one-task `openseeker-monid-plumbing-glm-20260921` run uses Alexandra GLM only
to validate the native environment path. Its result is not a Super-model eval.

The native Monid smoke passed with `ok=true` and no errors in
`artifacts/evals/search/openseeker-monid-plumbing-glm-20260921-r3/traces.jsonl`.
`configs/lumi/rl_nemotron_super_r5_128k_16t8i_sc_pilot_r30_search.toml` adds
OpenSeeker at family weight 1.0 and a 32-task BrowseComp-Plus eval every ten steps.
The native Slurm launcher dry-run passed; it has not been submitted.
OpenSeeker loading selects only question/answer columns to avoid decoding unused
training trajectories. Temporary native eval launchers must guard CLI invocation
with `if __name__ == "__main__"` for spawned workers; task metric hooks must be async.
