# TRUCE-Rec

This repository is TRUCE-Rec unless the user states otherwise. It is a
research-grade LLM4Rec codebase for uncertainty-aware generative
recommendation, with a shared data, baseline, OursMethod, evaluation, export,
and reproducibility scaffold.

Current repository identity:

- GitHub: `https://github.com/appleweiping/TRUCE-Rec.git`
- Historical remote alias in this checkout: `https://github.com/appleweiping/uncertainty-llm4rec.git`
- Local path: `D:\Research\TRUCE-Rec`
- Active branch: `main`
- Current stage: Gate R0 pre-experiment reviewer gate after Phases 1-7 passed
  and were pushed.

## Evidence Labels

Use these labels consistently:

- Smoke/mock: fixture-data or MockLLM runs that verify code paths only.
- Pilot: small approved real-data run used to debug the protocol.
- Diagnostic: prompt, grounding, candidate, or artifact QA; not a paper
  conclusion.
- Paper result: approved real experiment with tracked code, saved config, logs,
  raw outputs where applicable, predictions, metrics, and artifact checklist.

Strict rules:

- Smoke outputs are not paper evidence.
- MockLLM outputs are not paper evidence.
- Pilot/API diagnostic outputs are not paper conclusions.
- Ignored local diagnostics under `outputs/` or `data/processed/` are not
  paper evidence unless explicitly promoted by a later approved protocol.
- Formal paper results must come from approved real experiment configs, tracked
  code, saved configs, logs, raw outputs, predictions, and metrics.

## Current Status

Implemented and smoke-tested:

- dataset preprocessing and tiny fixture data;
- random, popularity, BM25, MF, sequential Markov, and LLM mock baselines;
- MockLLM, OpenAI-compatible provider interface, HF provider scaffold, and
  LoRA dry-run scaffold;
- unified prediction schema and shared evaluator;
- ranking, validity, confidence, calibration, coverage, diversity, novelty,
  long-tail, efficiency, slicing, aggregation, and table export support;
- Phase 6 OursMethod:
  `Calibrated Uncertainty-Guided Generative Recommendation`;
- Phase 7 paper support, real experiment templates, reproduction docs, and
  safe preflight helpers.

Not yet completed:

- no approved paper-result experiment suite;
- no claim that OursMethod is effective;
- no new real API run in Gate R0;
- no HF model download;
- no real LoRA/QLoRA training;
- no final paper conclusions.

## Key Commands

Run smoke baselines:

```powershell
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_all_baselines.yaml
```

Run Phase 6 smoke suite:

```powershell
.\.venv\bin\python.exe scripts\run_all.py --config configs/experiments/smoke_phase6_all.yaml
```

Run all tests:

```powershell
.\.venv\bin\python.exe -m pytest
```

Validate a real experiment template without running it:

```powershell
.\.venv\bin\python.exe scripts\validate_experiment_ready.py --config configs/experiments/real_ours_method_template.yaml
```

List required artifacts for a planned run:

```powershell
.\.venv\bin\python.exe scripts\list_required_artifacts.py --config configs/experiments/real_ours_method_template.yaml
```

## Important Docs

- `AGENTS.md`: engineering and research governance rules.
- `docs/RESEARCH_IDEA.md`: core research direction.
- `docs/experiment_protocol.md`: split, candidate, prompt, LLM, metric, and
  leakage protocol.
- `docs/real_experiment_matrix.md`: planned real experiment groups.
- `docs/reproduction.md`: local reproduction commands.
- `docs/pre_experiment_checklist.md`: real-run readiness checklist.
- `docs/result_artifact_checklist.md`: artifact contract.
- `docs/baselines.md`: baseline readiness and limitations.
- `docs/ours_method_plan.md`: Phase 6 method plan and boundaries.
- `docs/ablation_protocol.md`: OursMethod ablation protocol.
- `docs/leakage_fairness_checklist.md`: leakage/fairness safeguards.
- `docs/server_runbook.md`: API/HF/server/LoRA safety runbook.

## Package Layout

The active implementation package is `src/llm4rec/`. Older `storyflow`
references in historical local notes are not the active Phase 6/7 package
contract.

## Real Experiment Rule

Before any real pilot, fill a `configs/experiments/real_*_template.yaml`,
validate it with `scripts/validate_experiment_ready.py`, confirm datasets and
resources, and keep the safety flags blocking API calls, downloads, and
training until the user explicitly approves the run.
