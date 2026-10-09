# SQL training and agentic evaluation

The r16 pilot config adds `text2sql` from danish-foundation-models/text2sql-env, branch `codex/verifiers-v1-workspace-harness`, commit `227a5fd`, exposed through `deps/text2sql-env`. Its current-verifiers compatibility patch replaces the ignored class-level tools tuple with `toolsets(config)` and reports execution accuracy separately from shaped reward. Keep the SQL source and patch with the run; a package name alone is insufficient provenance.

BIRD train has 9,428 questions over 69 databases. Dev has 1,534 questions over 11 disjoint databases. Both load in full from `/scratch/project_465002751/oliver/data/bird`, currently a shared read-only input dependency, not a bench-owned copy. Use all train questions, not the earlier Qwen-filtered derivatives. Train receives one equal source weight. The existing agentic harness exposes four read-only SQL tools, no general shell, and runs in a local subprocess. Ten turns; Nemotron's renderer remains selected. SQL reward is not execution accuracy: runs-but-wrong receives 0.1.

Pilot evaluation schedule (one rollout per question):

| Eval | Questions | Interval |
| --- | ---: | ---: |
| BIRD dev | 64 | 10 |
| OpenThoughts TBLite | 16 | 10 |
| SWE-bench Verified | 32 | 20 |
| SWE-bench Multilingual | 24 | 20 |
| Terminal-Bench 2 | 16 | 20 |
| Senior SWE-Bench, GLM-5.3-Flash judge | 10 | 20 |

These are pilot monitoring samples, not full leaderboard evaluations. Existing three four-question SWE holdouts remain. For full endpoint evals use num_examples=-1 with the same harness, limits, data revision and judge settings; publish those settings with the scores. Check overlap with training task identities before interpreting benchmark results as uncontaminated generalization.

Senior SWE-Bench enables upstream judge stages using `ALEXANDRA_API_KEY` and `https://inference.alexandra.dk/v1`. The rubric/taste judges, classifier and validation-agent model are explicitly overridden to `openai/glm-5.3-flash`. Results are a judge-substituted protocol, not directly comparable to upstream default judges. Never silently disable judges: design-task scores otherwise become zero. The launch wrapper reads the existing key from prime-rl/.env; no key is embedded in configuration or artifacts.

Checks: 77 existing SQL tests plus the tool-discovery regression passed under the current runtime; full train/dev loading and database separation passed; r16 launcher dry-run passed. Dataset/image preloading and live model tool-call/sandbox validation are separate gates. The full-topology transfer benchmark uses another checkout and is unaffected. r16 is prepared, not submitted; promote the validated Simple transport separately after the full-topology test passes.
