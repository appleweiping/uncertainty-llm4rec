# Server Runbook

This runbook describes safe execution paths for smoke runs, API LLM runs, local
HF inference, and LoRA/QLoRA training. It does not authorize expensive jobs by
itself.

## Local smoke run

Safe local commands:

```powershell
python -m pytest
python scripts/run_all.py --config configs/experiments/smoke_phase6_all.yaml
python scripts/run_all.py --config configs/experiments/smoke_ours_method.yaml
```

These use fixture data and MockLLM. They are not paper evidence.

## API LLM run

Required before execution:

- explicit user confirmation of provider, model, budget, rate limit, and
  concurrency;
- provider config with `requires_confirm: true`;
- API key in an environment variable, never in source files;
- dry-run validation with `scripts/validate_experiment_ready.py`;
- no target leakage checklist completed.

Provider config should record:

- provider name and model;
- base URL if OpenAI-compatible;
- API key environment variable name;
- retry and timeout policy;
- cache location and raw-output location;
- rate-limit and cost-tracking settings.

Safe preflight:

```powershell
python scripts/validate_experiment_ready.py --config configs/experiments/real_llm_api_template.yaml
```

Do not run real API calls from templates until the user replaces TBD fields and
confirms the job.

## HF local model run

Required fields:

- local `model_name_or_path`;
- `allow_download: false` unless the user explicitly permits download;
- device selection;
- batch size;
- max tokens and decoding settings;
- logprob availability flag when relevant.

Warnings:

- loading a local model may require substantial CPU/GPU memory;
- API-style token costs may not apply, but latency and memory should be logged;
- no model should be downloaded automatically in Phase 7 templates.

## LoRA/QLoRA dry-run

Dry-run validation should check:

- config loads;
- dataset paths are present or marked TBD;
- output directory is writable;
- checkpoint/output path is defined;
- `dry_run: true` or `requires_confirm: true`;
- no real training flag is enabled.

Expected manifest:

- config snapshot;
- planned command;
- expected artifacts;
- `model_training=false`;
- `is_experiment_result=false`.

## Real LoRA/QLoRA server run

Real training requires explicit user approval and server details:

- GPU model and memory;
- CUDA/driver/Python environment;
- model path or approved download policy;
- dataset path and checksum where possible;
- output checkpoint directory;
- resume/eval-only policy;
- seed list;
- logging and monitoring plan.

Required outputs:

- checkpoints or adapters;
- trainer state;
- logs;
- metrics;
- resolved config;
- environment summary;
- git commit hash;
- evaluation predictions and metrics.

Risks:

- out-of-memory failures;
- partial checkpoints;
- train/eval leakage;
- model or data license constraints;
- cost and queue-time overruns.

No real LoRA/QLoRA command should be executed by default from Phase 7 templates.

## Claim policy

Do not claim any API, HF, or server result unless the run directory contains
the required logs, raw outputs where applicable, metrics, config snapshot, and
commit hash. Smoke/mock outputs are infrastructure checks only.
