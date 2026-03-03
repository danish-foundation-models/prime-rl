# LUMI Guide

This page documents the LUMI-specific workflow in this repo.

Current scope:
- Single-node SFT via Slurm
- Single-node RL via Slurm
- ROCm container + writable overlay runtime

Out of scope on this page:
- Multi-node templates/workflows
- Generic (non-LUMI) deployment docs

## Runtime Layout

LUMI runs in this repo use:
- Base SIF container (read-only)
- Overlay virtualenv at `overlay_prime_rl_lumi/venv/prime-rl-lumi`
- Repo source mounted at `/workdir`
- Flash-backed caches

Main scripts:
- Overlay builder: `scripts/build_lumi_overlay_prime_rl.sh`
- SFT submit wrapper: `scripts/lumi_submit_sft.sh`
- RL submit wrapper: `scripts/lumi_submit_rl.sh`

## Prerequisites

- Access to LUMI project `project_465002183`
- This repo checked out on a filesystem visible from compute nodes
- Base container available at:
  `/pfs/lustref1/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif`

## One-Time Setup

Build the overlay:

```bash
./scripts/build_lumi_overlay_prime_rl.sh
```

Defaults (override with env vars if needed):
- `SIF`: base container path
- `OVERLAY_DIR`: `./overlay_prime_rl_lumi`
- `OVERLAY_ENV_NAME`: `prime-rl-lumi`

## Cache Policy

Use only:
- `HF_HOME`
- `UV_CACHE_DIR`

Submit wrappers default both to `/flash/project_465002183/.cache/...`.

## Quickstart

### SFT

Dry-run:

```bash
./scripts/lumi_submit_sft.sh --dry-run
```

Submit:

```bash
./scripts/lumi_submit_sft.sh
```

### RL

Dry-run:

```bash
./scripts/lumi_submit_rl.sh --dry-run
```

Submit:

```bash
./scripts/lumi_submit_rl.sh
```

Use a custom config:

```bash
CONFIG=$PWD/configs/lumi/rl_single_node.toml ./scripts/lumi_submit_rl.sh
```

## Config Rules (LUMI-Specific)

1. RL inference startup requires `[inference]` in the resolved RL config.
- If omitted, launcher logs: `No inference config specified, skipping starting inference server.`
- This is useful only when you intentionally run an external inference service.

2. Be careful with `toml_files` inheritance.
- In this repo, inherited configs can be surprising for some overrides (especially array/table-heavy sections).
- For LUMI experiments, prefer explicit top-level config files when changing env mixes or deployment shape.

3. RL wrapper defaults token client to false.
- `scripts/lumi_submit_rl.sh` sets `ORCH_USE_TOKEN_CLIENT=false` by default.
- Override with:
  - `ORCH_USE_TOKEN_CLIENT=true` to force `--orchestrator.use-token-client`
  - `ORCH_USE_TOKEN_CLIENT=false` to force `--no-orchestrator.use-token-client`

4. Current LUMI SFT template runs one process per node task.
- Template currently calls `torch.distributed.run --nproc-per-node=1`.
- If you need multi-GPU SFT in this path, template changes are required.

## Monitoring and Logs

Queue status:

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
```

Accounting:

```bash
sacct -j <jobid> --format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,NNodes,NodeList -P
```

For a given `output_dir`:
- Slurm stdout: `<output_dir>/job_<jobid>.log`
- Slurm stderr: `<output_dir>/job_<jobid>.err`
- RL launcher log: `<output_dir>/logs/rl.log`
- RL orchestrator: `<output_dir>/logs/orchestrator.stdout`
- RL inference: `<output_dir>/logs/inference.stdout`
- RL trainer stream: `<output_dir>/logs/trainer.stdout`
- RL rank logs: `<output_dir>/logs/trainer/rank_<rank>.log`
- Torchrun redirected logs: `<output_dir>/torchrun/...`

Useful tails:

```bash
tail -F <output_dir>/logs/rl.log
tail -F <output_dir>/logs/orchestrator.stdout
tail -F <output_dir>/logs/inference.stdout
tail -F <output_dir>/logs/trainer/rank_0.log
```

## Troubleshooting

`FATAL: SIF not found`
- Check `SIF` env var or default container path.

`FATAL: overlay env not found`
- Build overlay first: `./scripts/build_lumi_overlay_prime_rl.sh`
- Or set `OVERLAY_DIR` / `OVERLAY_ENV_NAME` correctly.

`No inference config specified, skipping starting inference server`
- Add `[inference]` to RL config unless you intentionally use external inference.

Orchestrator repeats `Inference server was not reached`
- Usually means inference was not started or failed early.
- Check:
  - `<output_dir>/logs/rl.log`
  - `<output_dir>/logs/inference.stdout`

`uv` warning about `tool.uv.extra-build-dependencies`
- Known warning in this environment; current LUMI submit path still runs.

## Known Baseline

Validated in this branch:
- Single-node SFT works with overlay runtime.
- Single-node RL works with local inference + trainer + orchestrator.
- CP smoke validation (`trainer.model.cp=2`) completed successfully in job `16427480` on March 3, 2026.
