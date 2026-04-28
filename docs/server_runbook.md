# Server Runbook

This is a server execution framework for future Storyflow / TRUCE-Rec work. It
does not record any completed server run.

## Status

No server experiment, Qwen3-8B inference, Qwen3-8B + LoRA training, large-scale
baseline, or full Amazon run has been executed by Codex in this repository.

## Server Responsibilities

Server hardware is expected for:

- Qwen3-8B full inference when local execution is too heavy;
- Qwen3-8B + LoRA training;
- large Amazon Reviews 2023 category preprocessing;
- large-scale SASRec, BERT4Rec, GRU4Rec, LightGCN, and generative baseline
  runs where feasible;
- long-running observation, simulation, and triage experiments.

Local execution remains responsible for repository editing, small tests,
documentation, small real-data sanity checks, and report generation.

## Expected Server Directory Layout

Future server scripts and configs should use:

```text
configs/server/
scripts/server/
outputs/
runs/
data/raw/
data/cache/
```

Raw data, caches, outputs, and runs must not be committed unless a future task
explicitly adds sanitized lightweight fixtures.

## Environment Requirements

Future server setup should record:

- git remote and commit hash;
- Python version and package lock or exact install commands;
- CUDA, driver, GPU type, and GPU memory;
- model source and revision;
- dataset source, local path, checksum where available, and license/access
  notes;
- seed values and deterministic settings where feasible.

API keys and tokens must be supplied through environment variables or server
secret management. They must not be written into source files, configs, logs, or
commits.

## Planned Server Workflow

Future runbooks should expand these placeholders into exact commands after the
scripts exist:

1. Clone or update the repository on the server.
2. Verify the repository is on `main` and at the intended commit.
3. Create or activate the Python environment.
4. Install dependencies from the project dependency file.
5. Place or download datasets according to the dataset manifest.
6. Run preprocessing with manifest output.
7. Run Qwen3-8B inference or Qwen3-8B + LoRA training with config snapshots.
8. Write logs, metrics, generated outputs, and run manifests under `runs/` or
   `outputs/`.
9. Copy only sanitized summaries, configs, logs, and metrics needed for local
   analysis.

## Required Run Artifacts

Every future server run should produce:

- command line;
- git commit hash;
- config snapshot;
- dataset manifest;
- environment summary;
- seed values;
- stdout/stderr log path;
- output JSONL or metric path;
- failure/resume status;
- notes on whether the run is synthetic, pilot, or full.

## Claim Policy

Do not state that a server run succeeded unless its log or artifact is present
and inspected. Do not convert planned commands into experimental results.
