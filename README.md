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
popularity buckets, dataset manifests, a MovieLens 1M downloader, preprocessing
and split utilities, processed-data validation, Phase 2A prompt construction,
mock provider support, no-API observation input/output flow, and pytest coverage
for those foundations. Phase 2B API observation framework dry-run support and
Amazon Reviews 2023 Beauty readiness gates are in place. Phase 2C observation
analysis and local run registry utilities are now available for mock/dry-run
schema sanity and future pilot analysis. Real API observation, model training,
simulation, and full experiment phases have not started.

MovieLens 1M has also been verified as a local real-data sanity path from a
manually placed `data/raw/movielens_1m/ml-1m.zip` archive. The small
`sanity_50_users` prepare run produced only ignored local data outputs under
`data/processed/`; it is a pipeline check, not an experimental result.

A user-approved DeepSeek smoke and 20-example MovieLens sanity pilot have been
executed with `deepseek-v4-flash`, `thinking.type=disabled`, cache/resume, 10
requests/minute, and max concurrency 1. Early diagnostics found local TLS CA
and reasoning-token issues; those were fixed by using `certifi`, raising
`max_tokens`, and disabling thinking mode for this short JSON observation task.
The pilot produced parsed and grounded predictions plus analysis artifacts
under ignored `outputs/` paths. This is a small pilot only, not a full run and
not paper evidence. No model, toy model, full experiment, or server run has
been executed. The mock observation pipeline is only a no-API sanity path and
must not be reported as model behavior. Synthetic fixture under
`tests/fixtures/` is only for unit tests and pipeline sanity checks; it is not
an experimental result. Any future result must come from tracked code,
reproducible configs, logs, and output manifests.

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
- `docs/data_validation.md`: processed dataset validation checks and command.
- `docs/dataset_matrix.md`: dataset roadmap from synthetic fixtures through
  Amazon Reviews 2023 full categories.
- `docs/observation_pipeline.md`: Phase 2A no-API generative observation flow.
- `docs/api_observation.md`: Phase 2B provider config, dry-run, cache, resume,
  parsing, and real-pilot guardrails.
- `docs/observation_analysis.md`: analysis reports, reliability data, risk
  cases, and local ignored run registry.
- `docs/amazon_reviews_2023.md`: Amazon Beauty readiness and full-run entry.
- `docs/codex_execution_protocol.md`: required workflow for each Codex task.
- `docs/change_requests/` and `docs/decision_log.md`: scoped research and
  data-route decisions.
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
- `storyflow.data`: MovieLens 1M reading, title cleaning, chronological sorting,
  interaction filtering, k-core filtering, popularity computation, per-user
  leave-last splits, rolling examples, and global chronological split.
- `storyflow.generation`: prompt templates for title-level next-item generation,
  self-verification, probability confidence, and forced JSON output.
- `storyflow.providers.mock`: deterministic no-API mock provider for scaffold
  tests and observation pipeline sanity checks.
- `storyflow.providers`: provider config loading, cache key utilities,
  dry-run provider, and conservative OpenAI-compatible adapter scaffold.
- `storyflow.observation_parsing`: strict JSON, fenced JSON, embedded JSON, and
  regex fallback parsing for generated title, yes/no, and confidence.
- `storyflow.observation`: JSONL input construction, mock observation runner,
  grounding integration, metrics, reports, and resume support.
- `storyflow.analysis`: observation analysis summaries, reliability diagram
  data, head/mid/tail slices, risk case extraction, and ignored run registry
  helpers.
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

Raw datasets belong under `data/raw/` and are gitignored. Interim files,
processed outputs, generated caches, and run artifacts are also gitignored by
default. Small processed test fixtures may be committed under `tests/fixtures/`
when they are created in a future phase.

Dataset configs live under `configs/datasets/`:

- `movielens_1m.yaml`: local real-data sanity dataset with automatic download.
- `amazon_reviews_2023_beauty.yaml`: Hugging Face entry for All_Beauty.
- `amazon_reviews_2023_digital_music.yaml`: local raw entry for Digital_Music.
- `amazon_reviews_2023_handmade.yaml`: local raw entry for Handmade_Products.
- `amazon_reviews_2023_health.yaml`: local raw entry for
  Health_and_Personal_Care.
- `amazon_reviews_2023_sports.yaml`: Hugging Face entry for
  Sports_and_Outdoors.
- `amazon_reviews_2023_video_games.yaml`: Hugging Face entry for Video_Games.
- `steam.yaml`: planned/server-scale placeholder until a verified source is
  selected.

The current active API pilot target is DeepSeek only. API keys must be provided
through environment variables such as `DEEPSEEK_API_KEY`. Do not commit `.env`,
sensitive API responses, or paid API caches. `.env.example` also lists future
provider variables for later multi-model work, but Qwen API/Kimi/GLM and
server Qwen3 remain future phases.

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

The API adapter uses `certifi` for TLS certificate verification. The `dev`
extra installs pytest for tests.

## Dataset Commands

Download MovieLens 1M:

```powershell
python scripts/download_datasets.py --dataset movielens_1m
```

This writes raw files under `data/raw/movielens_1m/` and a download manifest
under `data/interim/movielens_1m/`. These paths are ignored by git.

If network access or certificate validation blocks the automatic download,
place the official GroupLens archive at
`data/raw/movielens_1m/ml-1m.zip` and rerun the same command. The downloader
will reuse the existing archive, compute its MD5, and extract it.

Prepare MovieLens 1M with the default per-user leave-last-two split:

```powershell
python scripts/prepare_dataset.py --dataset movielens_1m
```

Prepare a small local real-data sanity subset:

```powershell
python scripts/prepare_dataset.py --dataset movielens_1m --max-users 50 --output-suffix sanity_50_users
```

Processed outputs are written under `data/processed/movielens_1m/<run>/`:

- `item_catalog.csv`
- `interactions.csv`
- `item_popularity.csv`
- `user_sequences.jsonl`
- `observation_examples.jsonl`
- `preprocess_manifest.json`

Alternative split policy:

```powershell
python scripts/prepare_dataset.py --dataset movielens_1m --split-policy global_chronological
```

Amazon Reviews 2023 configs are server/full-scale entries. The download script
can cache the Hugging Face Dataset Viewer parquet index and write Chinese
instructions, but it does not download the full Amazon data locally by default:

```powershell
python scripts/download_datasets.py --dataset amazon_reviews_2023_beauty
```

Use full Amazon categories only on a machine with enough disk, network, and
runtime budget, and record manifests/logs before making any experimental claim.
MovieLens 1M must remain a sanity check; after Phase 2A, at least one Amazon
Reviews 2023 category should be prepared as the first full-data pipeline.

Inspect Amazon Reviews 2023 Beauty readiness without full download:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run --sample-records 3
```

Amazon Beauty full prepare is server/big-disk oriented and has not been run:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run
```

Run a local Beauty sample prepare after raw JSONL files are present:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --sample-mode --max-records 5000 --output-suffix sample_5k --min-user-interactions 1 --user-k-core 1 --item-k-core 1 --min-history 1 --max-history 20
```

Sample outputs are written under `data/processed/amazon_reviews_2023_beauty/`
and are ignored by git. They are readiness artifacts only, not full processed
data and not paper evidence. Full prepare is guarded; it requires an explicit
`--allow-full` flag after the user approves a full run.

Validate a processed dataset before building observation inputs:

```powershell
python scripts/validate_processed_dataset.py --dataset movielens_1m --processed-suffix sanity_50_users
```

Validation outputs are written under `outputs/data_validation/...` and are
ignored by git.

## Phase 2A Mock Observation

Build prompt JSONL inputs from processed examples:

```powershell
python scripts/build_observation_inputs.py --dataset movielens_1m --processed-suffix sanity_50_users --split test --max-examples 20 --stratify-by-popularity
```

Run the no-API mock observation pipeline:

```powershell
python scripts/run_observation_pipeline.py --provider mock --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --max-examples 20
```

This writes ignored outputs under:

- `outputs/observation_inputs/...`
- `outputs/observations/mock/...`

The mock provider is deterministic and does not require API keys. It exists to
test prompt construction, response parsing, title grounding, correctness
labeling, metrics, and resume behavior. It is not a real API pilot and not a
paper result. The next phase is to add real provider adapters with explicit
rate limits, retries, cache/resume, token/cost accounting, and environment
variable API keys.

## Phase 2B API Framework Dry-Run

Provider configs live under `configs/providers/`. DeepSeek is the current
single-provider pilot target and is configured for OpenAI-compatible
`/chat/completions` with `thinking.type=disabled` so the smoke test requests a
short final JSON answer rather than reasoning-only output. Future providers
intentionally keep endpoint/model values as `TODO_CONFIRM...` placeholders
until their exact setup is confirmed.

Run a dry-run API observation:

```powershell
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --max-examples 5 --dry-run
```

Dry-run writes ignored outputs under `outputs/api_observations/...` and never
calls the network or reads API keys. A future real pilot requires
`--execute-api`, confirmed config fields, an environment variable key, and user
approval of provider, model, budget, and rate limits.

Check DeepSeek smoke-test readiness without making a network call:

```powershell
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --sample-size 5 --stage smoke --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 10 --approved-budget-label USER_APPROVED_SMOKE --execute-api-intended
```

This command never prints the API key value and never calls DeepSeek. It only
checks whether all required gates are satisfied.

Current status markers:

- Mock observation: ready for local sanity.
- API framework: dry-run ready.
- Observation analysis: ready for mock/dry-run schema sanity and future pilot
  analysis.
- DeepSeek API smoke: succeeded for 5 approved MovieLens sanity records after
  setting `thinking.type=disabled`.
- DeepSeek API pilot: succeeded for 20 approved MovieLens sanity records;
  outputs are ignored local pilot artifacts, not full results or paper
  evidence.
- Amazon Beauty local raw files: available for readiness/sample checks.
- Amazon Beauty sample prepare gate: implemented; sample outputs are ignored
  local artifacts, not full-data results.
- Amazon Beauty full processed data: not yet produced.

## Phase 2C Observation Analysis

Analyze a dry-run or mock observation directory:

```powershell
python scripts/analyze_observation.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_dry_run
```

Outputs are ignored by git and written under:

- `outputs/analysis/...`
- `outputs/run_registry/observation_runs.jsonl`

The analysis report includes reliability diagram data, head/mid/tail summaries,
wrong-high-confidence cases, correct-low-confidence cases, grounding failures,
parse failure summaries, and an exploratory popularity-confidence slope. When
the source run is mock or dry-run, these are only pipeline sanity artifacts and
must not be reported as real model behavior or paper evidence.

## Tests

Run the local test suite:

```powershell
python -m pytest
```

Current tests cover schema validation, title normalization, grounding,
calibration metrics, GroundHit, k-core filtering, interaction filtering,
chronological sorting, leave-last splits, rolling examples, popularity buckets,
Tail Underconfidence Gap, prompt construction, mock provider parsing,
observation input schema, grounding/correctness integration, metrics reporting,
and resume behavior. Tests use small committed fixtures only and do not
download data or call APIs.

## Basic Checks

Governance checks:

```powershell
Get-Location
git branch --show-current
git remote -v
git status --short --branch
```
