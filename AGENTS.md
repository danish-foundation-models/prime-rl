# AGENTS.md

## LUMI workspace

This checkout is the training runtime for the sibling bench at
`/scratch/project_465002751/rasmus`. Read the bench `AGENTS.md` before you work
on LUMI integration.

Start a LUMI task with these checks:

```bash
cd /scratch/project_465002751/rasmus/prime-rl
git status --short --branch
squeue -u "$USER" -o "%.18i %.24j %.10T %.10M %.6D %R"
```

Use this operational image:

```text
/scratch/project_465002751/shared/sif/lumi-posttrain-u24r71-t211-vllm024.sif
```

Run the image without `--rocm` and without an overlay. Keep PRIME-RL source and
bundled environments outside the image. The LUMI launchers put live source
before generated package metadata on `PYTHONPATH`.

Keep outputs under `/scratch/project_465002751/rasmus/artifacts/training`. Keep
caches under `/scratch/project_465002751/rasmus/.cache`. Set only
`HF_HOME=/scratch/project_465002751/.cache/huggingface` for Hugging Face cache
placement.

Dry-run before submission:

```bash
CONFIG=$PWD/configs/lumi/<config>.toml ./scripts/lumi_submit_sft.sh --dry-run
CONFIG=$PWD/configs/lumi/<config>.toml ./scripts/lumi_submit_rl.sh --dry-run
```

Remove `--dry-run` only after you inspect the rendered Slurm script and output
directory. See `docs/lumi.md` for runtime details.

## Writing code

- **Minimal try/except**: let errors propagate — silent failures hide bugs. Only catch exceptions for intentional fault tolerance (retries, robustness).
- **Don't touch `optimization_dtype` / `reduce_dtype`**: never change these model config fields (or their defaults in `trainer.py`) unless the user explicitly asks. They're load-bearing numerical knobs — flipping bfloat16/float32 silently changes training dynamics.
- **Targeted comments**: don't explain your work process or reference old code. Use targeted comments sparingly to clarify ambiguous logic.
- **Zen of Python**: remember the Zen of Python when writing code.
```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

## Running code

- **Always use uv**: run code with `uv run` or `uv run <command>`, never raw `python`.
- **Adding dependencies**: add to `pyproject.toml` and run `uv sync --all-extras` to install and lock them.
- **Git dependency pins**: when pinning git dependencies in `pyproject.toml`, always use a small (7-char) commit hash for the `rev` field.
- **Never edit `.venv/`**: the local virtual env is read-only. Edits there are silently overwritten by the next `uv sync` and don't propagate to teammates or CI. Reading files under `.venv/` to understand library behavior is fine; writing is not. To fix a dependency issue, update `pyproject.toml` (pin a fork via 7-char commit hash if needed), vendor the code into `src/`, or patch upstream.

## Docs

- **Docs reflect `main`, not history**: `docs/` describes the current state of the codebase only. Don't mention removed/legacy fields, migration paths, or "this used to be X" anecdotes.

## Skills

Skills live in `skills/` and are symlinked to `.claude/skills/`. They teach agents how to handle specific workflows (e.g. starting the inference server, writing configs). When you make changes to the codebase, check if any skills need to be updated to stay accurate.

You are responsible for maintaining the skills folder. When a workflow fails and you fix it – whether with help from the user or through trial and error – you must update the skills to make implicit knowledge explicit. You are also responsible for keeping the skills up to date whenever you or anyone else modifies the code.

Also update this `AGENTS.md`, `docs/lumi.md`, or the relevant README when the
fix changes a repository-wide command, path, runtime rule, or safety check. If
an instruction disagrees with tested behavior, correct the instruction in the
same change. Do not leave a known-bad workflow for the next agent to rediscover.

## Testing

Write tests as plain functions with pytest fixtures. Don't use class-based tests.

- **Conservative test additions**: don't add new tests unless the user explicitly asks for them or it's clearly necessary. Editing existing tests is fine, but adding new test files or test functions should be the exception, not the default.
- **Test what matters**: only test code with clear, isolated logic — pure functions, abstract base classes, data transformations, well-defined algorithms. Don't test runtime-level code, framework glue, or anything that requires extensive mocking/patching just to get a test to pass. If you need to patch everything out to make it testable, it's probably not worth testing.

## Git

- **Branch prefixes**: use `feat/`, `fix/`, `chore/`; use `exp/` for experiment branches (configs, run summaries, pins, notes).

## GitHub

- **Draft PRs**: always create PRs as drafts (`gh pr create --draft`) to avoid triggering CI unnecessarily.
- **Pull requests**: do not include a "test plan" section in PR descriptions unless you actually ran tests to verify the changes or the user explicitly asked for one.
- **Keep PR descriptions in sync**: every time you push commits to a PR, also update the PR description (`gh pr edit <num> --body-file ...`) so it reflects the current state of the branch — not just what was true when the PR was opened. Preserve any auto-generated blocks (e.g. `<!-- CURSOR_SUMMARY -->`).
