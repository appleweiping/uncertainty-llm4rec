# Storyflow / TRUCE-Rec

Storyflow / TRUCE-Rec is a research codebase for uncertainty-aware LLM-based
generative recommendation. The project studies whether an LLM that recommends
an item by generating its title also knows whether that recommendation is
correct, and whether its confidence reflects user utility rather than
popularity, familiarity, exposure bias, grounding ease, or training noise.

The active repository is:

- `https://github.com/appleweiping/uncertainty-llm4rec.git`

The active local project directory is:

- `D:\Research\TRUCE-Rec`

The active branch is:

- `main`

`Storyflow.md` is the conceptual source of truth. `AGENTS.md` is the execution
contract for Codex work in this repository.

## Current Status

Phase 0 governance/scaffold is established. A minimal Python research scaffold
now exists with schemas, transparent title grounding, basic calibration metrics,
popularity buckets, and pytest coverage for those foundations. Data download,
API observation, model training, simulation, and full experiment phases have
not started.

No data has been downloaded. No paid or external API has been called. No model,
toy model, pilot experiment, full experiment, or server run has been executed.
The synthetic fixture under `tests/fixtures/` is only for unit tests and
pipeline sanity checks; it is not an experimental result. Any future result
must come from tracked code, reproducible configs, logs, and output manifests.

## Scientific Scope

The core task is title-level generative recommendation:

1. The model receives a user interaction history as item titles and metadata.
2. The model generates one or more item titles.
3. Each generated title is grounded to a catalog item.
4. Correctness, confidence, calibration, popularity coupling, grounding
   uncertainty, and echo risk are evaluated together.
5. The framework stage targets Qwen3-8B + LoRA or a comparable small-model
   training setup, normally on server hardware.

This project is not a generic top-k ranking-only recommender, not a simple
prompting demo, and not a place for fabricated tables, metrics, or claims.

## Project Documents

- `Storyflow.md`: conceptual research specification and thesis.
- `AGENTS.md`: repository rules and Codex operating contract.
- `docs/implementation_plan.md`: phased engineering and research plan.
- `docs/experiment_protocol.md`: task definition, observation protocol, metrics,
  and local/server split.
- `docs/codex_execution_protocol.md`: required workflow for each Codex task.
- `docs/server_runbook.md`: server execution framework. It is a scaffold only;
  no server run has been performed by Codex.
- `references/README.md`: policy for local reference material such as
  `recprefer.zip`.

## Python Scaffold

The repository uses a `src/` Python package layout. The following package
directories are present:

```text
src/storyflow/
src/storyflow/data/
src/storyflow/grounding/
src/storyflow/generation/
src/storyflow/providers/
src/storyflow/confidence/
src/storyflow/metrics/
src/storyflow/analysis/
src/storyflow/simulation/
src/storyflow/triage/
src/storyflow/models/
src/storyflow/training/
src/storyflow/baselines/
src/storyflow/utils/
tests/
```

Implemented foundation modules:

- `storyflow.schemas`: item catalog, interaction, user sequence, generative
  prediction, grounded prediction, confidence, and observation example records.
- `storyflow.grounding`: title normalization, exact match, normalized exact
  match, stdlib fuzzy match, grounding score, and ambiguity placeholder.
- `storyflow.metrics`: ECE, Brier score, CBU_tau, WBC_tau, GroundHit,
  popularity bucket assignment, and Tail Underconfidence Gap.
- `tests/fixtures/`: synthetic records used only for tests.

The remaining subpackages are intentionally lightweight placeholders for later
phases.

## Milestones

- Phase 0: governance/scaffold.
- Phase 1: data download and preprocessing.
- Phase 2: API observation pipeline.
- Phase 3: full observation and baselines.
- Phase 4: CURE/TRUCE framework.
- Phase 5: echo simulation and data triage.
- Phase 6: full experiments and paper artifacts.

See `docs/implementation_plan.md` for acceptance criteria.

## Local Versus Server Work

Local work is for repository editing, preprocessing where feasible, synthetic
tests, small real-data sanity checks, API-based observation after explicit
configuration, report generation, and plotting from completed outputs.

Server work is expected for Qwen3-8B full inference, Qwen3-8B + LoRA training,
large Amazon categories, large baselines, and long-running experiments. Codex
cannot access the remote server and must not claim server results unless the
user provides logs or artifacts.

## Data, API, and Reference Policy

Raw datasets belong under `data/raw/` and are gitignored. Generated caches,
outputs, and run artifacts are also gitignored by default. Small processed test
fixtures may be committed under `tests/fixtures/` when they are created in a
future phase.

API keys must be provided through environment variables such as
`DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `MOONSHOT_API_KEY`, and
`ZHIPUAI_API_KEY`. Do not commit `.env`, sensitive API responses, or paid API
caches.

Large reference files such as PDFs and `recprefer.zip` must remain local under
`references/` and are not committed. Commit only lightweight indexes and notes.

## Installation

For local development:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Some MSYS2-style Python builds create `.venv\bin\python.exe` instead of
`.venv\Scripts\python`; use the path that exists in your local environment.

The current runtime code has no required third-party dependency. The `dev`
extra installs pytest for tests.

## Tests

Run the local test suite:

```powershell
python -m pytest
```

Current tests cover schema validation, title normalization, grounding,
calibration metrics, GroundHit, popularity buckets, and Tail Underconfidence
Gap. These tests use synthetic fixtures only and do not download data or call
APIs.

## Basic Checks

Governance checks:

```powershell
Get-Location
git branch --show-current
git remote -v
git status --short --branch
```
