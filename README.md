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
schema sanity and approved API analysis. A scoped DeepSeek Amazon Beauty
no-repeat full-slice API observation has been executed, but multi-provider/full
experiment, model training, simulation, and paper-result phases have not
started.
Qwen3-8B server observation scaffolding is now available as a plan/contract
layer under `configs/server/`, `scripts/server/`, and `storyflow.server`; no
Qwen3 inference, server run, or LoRA training has been executed.
The first CURE/TRUCE framework scaffold is also present under
`storyflow.confidence`: it defines exposure-counterfactual confidence features,
deterministic risk/echo scoring, and a reranking contract around
`C(u, i) ~= P(user accepts item i | user u, do(exposure=1))`. This is tested
scaffold code only. A feature builder now converts existing grounded
observation JSONL into this schema with manifests. Split-audited histogram
calibration and popularity residualization scaffolds record fit/eval provenance
for feature JSONL files. A deterministic reranker can now consume raw,
calibrated, or residualized feature rows and write a provenance manifest; none
of these are learned model results and no method result is claimed.
Processed-dataset audit tooling now checks repeat-target cases, chronological
split integrity, title quality, and head/mid/tail coverage before scaling API
observation.

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
not paper evidence.

After Amazon Beauty sample readiness, a user-approved 30-example DeepSeek
Amazon Beauty sample pilot was executed with `deepseek-v4-flash`,
`thinking.type=disabled`, cache enabled, explicit `execution_mode=execute_api`,
30 requests/minute, and max concurrency 3. The run produced parsed and grounded
pilot artifacts under ignored `outputs/` paths and was followed by analysis and
case-review diagnostics. It is a small sample pilot only, not a full Amazon run
and not paper evidence.

Amazon Reviews 2023 Beauty full preprocessing has now been executed locally
from user-provided raw JSONL files with `--allow-full` and validated as a data
readiness artifact. The ignored processed output is under
`data/processed/amazon_reviews_2023_beauty/full/` with 357 users, 479 catalog
items, 3315 interactions, and 2244 rolling observation examples after the
configured filtering. Full test observation inputs have been built under
ignored `outputs/observation_inputs/amazon_reviews_2023_beauty/full/`. This is
not an experiment result and not server evidence.

An approved DeepSeek Amazon Beauty no-repeat full-slice observation has been
executed on 185 repeat-free test inputs with `deepseek-v4-flash`, cache/resume,
`--run-stage full`, 30 requests/minute, and max concurrency 3. Raw/parsed/
grounded outputs, analysis, and case review are under ignored `outputs/` paths.
This is a scoped API observation artifact, not a paper conclusion; current
diagnostics indicate the next engineering gate should be retrieval-context or
catalog-constrained grounding, not a larger free-form run.

A follow-up user-approved DeepSeek retrieval-context diagnostic has also been
executed on the same 185 repeat-free test examples. After an initial local
network-permission failure that produced provider-stage failures and no raw
responses, the retry completed with cache/resume, 30 requests/minute, max
concurrency 3, and zero failed cases. It raised GroundHit from the free-form
slice's `0.173` to `0.973`, while target correctness remains uninterpretable as
recommendation accuracy because the diagnostic candidate set excludes the
held-out target by design. This artifact is evidence about prompt/candidate/
grounding behavior, not a paper result or method claim. The analysis layer now
adds candidate-set diagnostics for this prompt family: on the same ignored
retrieval-context artifact, `162/185` grounded outputs selected a provided
candidate title, all `185/185` examples excluded the held-out target, and the
mean selected candidate rank was `5.586`. These are scope/QA diagnostics for
candidate-prompt behavior, not recommendation-accuracy evidence.

A matching user-approved catalog-constrained diagnostic has been executed on
the same 185 repeat-free examples with target-excluding round-robin candidates.
It completed with zero failed API cases and wrote ignored raw/parsed/grounded,
analysis, case-review, and candidate-diagnostic artifacts. Unlike the
retrieval-context gate, the catalog-constrained prompt produced GroundHit
`0.784`, `40/185` ungrounded low-confidence outputs, and `140/185` grounded
outputs that selected a provided candidate. Because all `185/185` candidate
sets exclude the held-out target, this remains prompt/candidate/grounding QA
and must not be treated as recommendation accuracy or a method result.

No model, toy model, full experiment, or server run has been executed. The mock
observation pipeline is only a no-API sanity path and must not be reported as
model behavior. Synthetic fixture under
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
- `docs/pilot_case_review.md`: pilot case-review and failure-taxonomy layer for
  prompt, grounding, and confidence triage before scaling.
- `docs/baseline_observation.md`: lightweight popularity, train-split
  co-occurrence, and ranking-to-title baseline observation interface.
- `docs/cure_truce_framework.md`: exposure-counterfactual confidence feature
  schema, deterministic CURE/TRUCE scoring, calibration, and popularity
  residualization scaffolds.
- `docs/grounding_diagnostics.md`: catalog duplicate-title and low-margin
  grounding diagnostics before API scale-up.
- `docs/amazon_reviews_2023.md`: Amazon Beauty readiness and full-run entry.
- `docs/codex_execution_protocol.md`: required workflow for each Codex task.
- `docs/change_requests/` and `docs/decision_log.md`: scoped research and
  data-route decisions.
- `docs/reviews/`: periodic self-review artifacts for research-quality,
  non-toy, and reviewer-risk checks.
- `docs/server_runbook.md`: server execution framework. It is a scaffold only;
  no server run has been performed by Codex.
- `configs/server/qwen3_8b_observation.yaml`: Qwen3 server observation
  contract. It is a plan/config scaffold only, not a completed inference run.
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
src/storyflow/server/
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
  self-verification, probability confidence, forced JSON output, and a
  catalog-constrained diagnostic prompt for grounding-gate checks.
- `storyflow.providers.mock`: deterministic no-API mock provider for scaffold
  tests and observation pipeline sanity checks.
- `storyflow.providers`: provider config loading, cache key utilities,
  dry-run provider, and conservative OpenAI-compatible adapter scaffold.
- `storyflow.observation_parsing`: strict JSON, fenced JSON, embedded JSON, and
  regex fallback parsing for generated title, yes/no, and confidence.
- `storyflow.observation`: JSONL input construction, mock observation runner,
  grounding integration, metrics, reports, and resume support.
- `storyflow.analysis`: observation analysis summaries, reliability diagram
  data, head/mid/tail and repeat-target slices, candidate-prompt diagnostics,
  risk case extraction, and ignored run registry helpers.
- `storyflow.analysis.grounding_diagnostics`: catalog duplicate normalized
  title checks and optional grounding candidate margin audits.
- `storyflow.baselines`: lightweight popularity and co-occurrence title
  baselines plus a ranking-JSONL adapter that write the same grounded
  observation schema.
- `storyflow.confidence`: exposure-counterfactual confidence feature schema,
  grounded-observation feature builder, popularity residual/deconfounding,
  echo-risk/risk components, deterministic CURE/TRUCE score, reranking
  scaffold, split-audited histogram calibration scaffold, split-audited
  popularity residualization scaffold, and a JSONL reranking contract that can
  consume calibrated/residualized confidence proxies. This is not a trained
  calibrator, reranker, or result.
- `storyflow.server`: Qwen3-8B server observation plan/execution contract that
  mirrors API observation output layers while defaulting to plan-only mode.
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

DeepSeek readiness gates now distinguish `smoke`, `pilot`, and `full` stages.
Full-stage API observation still requires an explicit provider/model/rate/
concurrency/budget manifest, `--execute-api`, and `--run-stage full`;
raw/parsed/grounded outputs remain under ignored `outputs/`.

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

Amazon Beauty full prepare has been run locally from the current raw JSONL
files and remains ignored by git. Re-run only when raw data, preprocessing
config, or code changes require regeneration:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --reviews-jsonl data/raw/amazon_reviews_2023_beauty/All_Beauty.jsonl --metadata-jsonl data/raw/amazon_reviews_2023_beauty/meta_All_Beauty.jsonl --output-suffix full --allow-full
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

Run the deeper processed observation audit when a validation warning or a
scale-up decision needs split-level evidence:

```powershell
python scripts/audit_processed_dataset.py --dataset amazon_reviews_2023_beauty --processed-suffix full
```

Audit outputs are written under `outputs/data_audits/...` and include
`dataset_audit_summary.json`, `dataset_audit_report.md`,
`repeated_target_cases.jsonl`, and `duplicate_history_cases.jsonl`. Repeated
target-in-history examples are warnings, not automatic blockers: in e-commerce
they may be repeat purchase behavior or duplicate review artifacts, so future
API/full reports should stratify them before making paper claims.

Build repeat-aware observation inputs for sensitivity analysis before a full
API run:

```powershell
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --stratify-by-popularity --repeat-target-policy exclude
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --stratify-by-popularity --repeat-target-policy only
```

These write ignored files such as `test_no_repeat_forced_json.jsonl` and
`test_repeat_only_forced_json.jsonl`. They do not call an API. Full reports
should compare all / no-repeat / repeat-only slices when repeated purchases or
duplicate reviews are present.

## Phase 2A Mock Observation

Build prompt JSONL inputs from processed examples:

```powershell
python scripts/build_observation_inputs.py --dataset movielens_1m --processed-suffix sanity_50_users --split test --max-examples 20 --stratify-by-popularity
```

Build a target-leakage-safe catalog-constrained grounding diagnostic input:

```powershell
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --split test --max-examples 30 --stratify-by-popularity --prompt-template catalog_constrained_json --candidate-count 20
```

The constrained prompt is a debug gate for catalog coverage, title
normalization, parser behavior, and grounding. It is not the main free-form
generative recommendation setting. By default the target item is excluded from
the candidate list, so correctness from this constrained file is not
interpretable as recommendation accuracy.

Build the no-API Amazon Beauty observation gate variants in one command:

```powershell
python scripts/build_observation_gate_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --split test --max-examples 30 --stratify-by-popularity --candidate-count 20
```

This writes free-form `forced_json`, round-robin `catalog_constrained_json`,
and history-token-overlap `retrieval_context_json` input files plus a gate
manifest under `outputs/observation_inputs/...`. Gate files are named
`test_gate30_*.jsonl` so they do not overwrite the full split input such as
`test_forced_json.jsonl`. The retrieval-context prompt
uses catalog titles as leakage-safe grounding context only; it does not call an
API and does not produce a model result.

For the Amazon Beauty full no-repeat slice after free-form grounding failures,
build the leakage-safe comparison gate without calling an API:

```powershell
python scripts/build_observation_gate_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --max-examples 185 --stratify-by-popularity --candidate-count 20 --repeat-target-policy exclude
```

This writes ignored `test_gate185_no_repeat_*` inputs. The constrained variants
exclude the held-out target by default, so they are prompt/grounding diagnostics
and not recommendation-accuracy evidence.

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
approval of provider, model, budget, rate limits, and concurrency. Cache keys
include `execution_mode`, so dry-run cache records cannot be reused as real API
responses.

Check DeepSeek smoke-test readiness without making a network call:

```powershell
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --sample-size 5 --stage smoke --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 10 --approved-max-concurrency 1 --approved-budget-label USER_APPROVED_SMOKE --execute-api-intended
```

This command never prints the API key value and never calls DeepSeek. It only
checks whether all required gates are satisfied.

For an explicitly approved faster Amazon Beauty sample pilot, use a separate
budget label and keep cache/resume enabled:

```powershell
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --sample-size 30 --stage pilot --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 30 --approved-max-concurrency 3 --approved-budget-label USER_APPROVED_BEAUTY_SAMPLE30_PARALLEL --execute-api-intended --allow-over-20
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --output-dir outputs/api_observations/deepseek/amazon_reviews_2023_beauty/sample_5k/test_forced_json_api_pilot30_parallel --max-examples 30 --execute-api --rate-limit 30 --max-concurrency 3 --run-label amazon_beauty_sample30_deepseek_parallel --budget-label USER_APPROVED_BEAUTY_SAMPLE30_PARALLEL
```

These commands are for approved pilot work only. They do not authorize a paper
claim or a full provider sweep.

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
- DeepSeek Amazon Beauty sample pilot: succeeded for 30 approved sample records
  with max concurrency 3 and 30 requests/minute; outputs are ignored local
  pilot artifacts, not a full Amazon/API run and not paper evidence.
- Amazon Beauty local raw files: available for readiness/sample/full prepare
  checks.
- Amazon Beauty sample prepare gate: implemented; sample outputs are ignored
  local artifacts, not full-data results.
- Amazon Beauty full processed data: produced locally and validated as a
  readiness artifact; ignored under `data/processed/...`, not paper evidence.
- DeepSeek Amazon Beauty retrieval-context diagnostic: executed for the
  185-example repeat-free full slice with zero failed cases. GroundHit improved
  substantially versus the free-form path, but target-hit correctness is not a
  recommendation metric because the held-out target is excluded from diagnostic
  candidates. Candidate diagnostics have been added to the analysis layer for
  this run: `162/185` grounded outputs selected one of the provided candidate
  titles, target leakage was `0/185`, and selected candidate buckets were
  head=49, mid=73, tail=40. This remains prompt/candidate QA, not a paper
  result.
- DeepSeek Amazon Beauty catalog-constrained diagnostic: executed for the same
  185 repeat-free examples with round-robin target-excluding candidates. The
  run had zero provider/parse failures, GroundHit `0.784`, candidate selection
  `140/185`, target leakage `0/185`, selected candidate buckets head=29,
  mid=40, tail=71, and `40/185` ungrounded low-confidence cases. This is a
  prompt/candidate QA artifact, not recommendation accuracy.

## Phase 2C Observation Analysis

Analyze a dry-run or mock observation directory:

```powershell
python scripts/analyze_observation.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_dry_run
```

Outputs are ignored by git and written under:

- `outputs/analysis/...`
- `outputs/run_registry/observation_runs.jsonl`

The analysis report includes reliability diagram data, head/mid/tail summaries,
repeat-target slices, wrong-high-confidence cases, correct-low-confidence cases,
grounding failures, parse failure summaries, candidate-prompt diagnostics, and
an exploratory popularity-confidence slope. It writes `repeat_summary.json` and
`candidate_diagnostic_summary.json`, so Amazon
Beauty runs can compare all / no-repeat / repeat-only behavior without mixing
repeat purchase or duplicate-review diagnostics into ordinary next-item claims.
When the source run is mock or dry-run, these are only pipeline sanity
artifacts and must not be reported as real model behavior or paper evidence.
For retrieval-context or catalog-constrained diagnostic prompts, the candidate
summary reports generated-in-candidate-set rate, selected rank/score/bucket,
target-leak status, and history-item copying. If the held-out target is
excluded from candidates, target-hit correctness is not recommendation
accuracy.

Compare completed prompt/grounding gate analyses on the same input slice:

```powershell
python scripts/compare_observation_runs.py --run free_form=outputs/analysis/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_api_full185_20260429/analysis_summary.json,outputs/case_reviews/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_api_full185_20260429/case_review_summary.json --run retrieval_context=outputs/analysis/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_retrieval_context_json_c20_api_full185_retry_20260429/analysis_summary.json,outputs/case_reviews/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_retrieval_context_json_c20_api_full185_retry_20260429/case_review_summary.json --run catalog_constrained=outputs/analysis/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_catalog_constrained_json_c20_api_full185_20260429/analysis_summary.json,outputs/case_reviews/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_catalog_constrained_json_c20_api_full185_20260429/case_review_summary.json --output-dir outputs/analysis_comparisons/deepseek/amazon_reviews_2023_beauty/full/full185_prompt_gates_20260429 --source-label deepseek-beauty-full185-prompt-gates
```

This comparison reads only existing analysis/case-review summaries and writes
ignored JSONL/CSV/Markdown artifacts. It is a prompt/candidate/grounding QA
view, not a paper-result table or recommendation-accuracy comparison when
diagnostic candidate sets exclude the held-out target.

## Qwen3-8B Server Observation Scaffold

Qwen3-8B observation is server-oriented. The committed scaffold prepares an
API-compatible output contract and command plan without loading a model:

```powershell
python scripts/server/run_qwen3_observation.py --config configs/server/qwen3_8b_observation.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl --output-dir outputs/server_observations/qwen3_8b/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_plan --max-examples 20 --run-label qwen3_beauty_plan
```

The default command writes `request_records.jsonl`,
`expected_output_contract.json`, `server_command_plan.md`, and `manifest.json`
under ignored `outputs/`. It records `api_called=false`,
`server_executed=false`, `model_inference_run=false`, and
`is_experiment_result=false`.

Actual server inference is guarded by an explicit flag and must be run only
after the server environment, model source, input slice, output path, and
artifact-return policy are approved and logged:

```powershell
python scripts/server/run_qwen3_observation.py --config configs/server/qwen3_8b_observation.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl --output-dir outputs/server_observations/qwen3_8b/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_server --execute-server --run-stage full --run-label qwen3_beauty_full_server
```

When executed on a server with `torch` and `transformers`, the script writes
the same observation layers as the API runner: raw responses, parsed
predictions, failed cases, grounded predictions, metrics, report, and manifest.
Codex has not run this command locally or on a server.

Review concrete pilot cases and failure taxonomy:

```powershell
python scripts/review_observation_cases.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_api_pilot20_non_thinking_20260428
```

This writes ignored artifacts under `outputs/case_reviews/...`, including
`case_review_summary.json`, `case_review_cases.jsonl`, and `case_review.md`.
The review joins generated titles, grounded catalog items, target titles,
confidence, popularity buckets, and the tail of the user's history. It also
writes machine-readable `recommended_actions` per case and
`recommended_next_actions` in the summary, so pilot review can point to prompt,
grounding, self-verification, popularity-residual, or tail-calibration follow
ups. It is a pilot diagnostic, not a paper result.

Audit catalog grounding ambiguity before scaling API calls:

```powershell
python scripts/analyze_grounding_diagnostics.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k
```

To include an existing pilot's grounded predictions:

```powershell
python scripts/analyze_grounding_diagnostics.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --grounded-jsonl outputs/api_observations/deepseek/amazon_reviews_2023_beauty/sample_5k/test_forced_json_api_pilot30_parallel_20260428/grounded_predictions.jsonl --manifest-json outputs/api_observations/deepseek/amazon_reviews_2023_beauty/sample_5k/test_forced_json_api_pilot30_parallel_20260428/manifest.json
```

This writes ignored artifacts under `outputs/grounding_diagnostics/...`,
including duplicate normalized title groups, low-margin grounding cases, and
`grounding_failure_cases.jsonl`. Failure cases are tagged as near-miss
candidates, weak catalog overlap, no catalog support, generic generated title,
duplicate-title risk, and high-confidence ungrounded cases, with recommended
next actions for prompt, catalog, or grounding review. It is a QA artifact
only, not a model-performance result.

## Amazon Beauty Sample Observation Gate

After a local Amazon Beauty sample prepare exists, validate it before building
observation inputs:

```powershell
python scripts/validate_processed_dataset.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k
```

Then build a small, stratified observation input file without calling any API:

```powershell
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --split test --max-examples 30 --stratify-by-popularity
```

The output is ignored under:

```text
outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl
```

This gate proves the Amazon Beauty sample is ready for prompt construction and
future approved API observation. It is not a full Amazon run and not paper
evidence.

If pilot case review shows many ungrounded high-confidence titles, build the
separate catalog-constrained and retrieval-context diagnostic inputs before
spending more API budget:

```powershell
python scripts/build_observation_gate_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --split test --max-examples 30 --stratify-by-popularity --candidate-count 20
python scripts/build_observation_gate_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --max-examples 185 --stratify-by-popularity --candidate-count 20 --repeat-target-policy exclude
```

This produces ignored inputs under:

```text
outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_gate30_forced_json.jsonl
outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_gate30_catalog_constrained_json_c20.jsonl
outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_gate30_retrieval_context_json_c20.jsonl
outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_observation_gate_manifest.json
```

## Baseline Observation

Run lightweight non-LLM baselines through the same generated-title grounding and
metrics schema:

```powershell
python scripts/run_baseline_observation.py --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline popularity --max-examples 30
python scripts/run_baseline_observation.py --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline cooccurrence --max-examples 30
python scripts/run_baseline_observation.py --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline ranking_jsonl --ranking-jsonl outputs/baseline_rankings/sasrec/sample_5k/test_rankings.jsonl --max-examples 30 --strict-ranking
```

Implemented baselines:

- `popularity`: most popular unseen catalog title with a normalized popularity
  confidence proxy.
- `cooccurrence`: train-split item co-occurrence over observation examples,
  with popularity fallback.
- `ranking_jsonl`: converts externally produced ranked item IDs, such as a
  future SASRec/BERT4Rec/GRU4Rec/LightGCN run, into the same title-level
  baseline schema. The adapter filters history items, looks up the selected
  catalog title, grounds it, and records non-calibrated rank/score confidence
  metadata.

Validate any external ranking artifact before adapting it:

```powershell
python scripts/validate_baseline_artifact.py --ranking-jsonl outputs/baseline_rankings/sasrec/sample_5k/test_rankings.jsonl --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline-family sasrec --model-family SASRec --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --split test --trained-splits train --strict
```

The validator writes an ignored manifest under
`outputs/baseline_artifact_validation/...` and checks input coverage, ranking
schema, score metadata, catalog item IDs, history overlap, and split/provenance
declarations. It does not run the ranker, call APIs, train models, download
data, or create a paper result.

These baselines do not call APIs, do not read keys, and do not train models.
They are sanity/reviewer-proofing artifacts until a full baseline protocol run
is explicitly executed and documented.

## CURE/TRUCE Framework Scaffold

The first framework code is a local, deterministic scaffold for
exposure-counterfactual confidence:

```text
C(u, i) ~= P(user accepts item i | user u, do(exposure=1))
```

`storyflow.confidence.ExposureConfidenceFeatures` keeps verbal confidence,
generation evidence, grounding confidence/ambiguity, popularity pressure,
history alignment, novelty, and observation labels separate. The CURE/TRUCE
score combines estimated exposure confidence, preference evidence, information
gain, grounding/overclaim risk, and echo risk, then emits a heuristic action:
`recommend`, `diversify`, `explore`, or `abstain`.

This scaffold does not call APIs, train a model, run Qwen3, or prove the
framework. It is a tested contract for later calibrators, rerankers, server
training, and triage modules.

Build CURE/TRUCE feature records from an existing grounded observation output:

```powershell
python scripts/build_confidence_features.py --grounded-jsonl outputs/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_api_full185_20260429/grounded_predictions.jsonl --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl --catalog-csv data/processed/amazon_reviews_2023_beauty/full/item_catalog.csv
```

This writes ignored `features.jsonl` and `manifest.json` under
`outputs/confidence_features/...` by default. If no catalog is supplied, the
builder does not substitute target popularity for wrong generated items; it
marks generated-item popularity as unknown so later calibrators cannot absorb a
target-popularity leak.

Fit and apply the split-audited calibration scaffold on feature rows that
contain a proper train/evaluation split:

```powershell
python scripts/calibrate_confidence_features.py --features-jsonl outputs/confidence_features/<source-run>/features.jsonl --fit-splits train --eval-splits validation,test --n-bins 10
```

This writes ignored `calibrated_features.jsonl` and `manifest.json` under
`outputs/confidence_calibration/...` by default. The command refuses fit/eval
split overlap unless `--allow-same-split-eval` is explicitly passed for a
diagnostic, records `api_called=false` and `model_training=false`, and must not
be interpreted as a learned CURE/TRUCE result.

Fit and apply the split-audited popularity residual scaffold on the same feature
rows:

```powershell
python scripts/residualize_confidence_features.py --features-jsonl outputs/confidence_features/<source-run>/features.jsonl --fit-splits train --eval-splits validation,test
```

This writes ignored `popularity_residualized_features.jsonl` and
`manifest.json` under `outputs/confidence_residuals/...` by default. The
scaffold fits a popularity-bucket mean confidence baseline on the requested fit
split and applies it only to evaluation splits, producing
`popularity_residual_confidence` and a recentered
`deconfounded_confidence_proxy`. If generated-item popularity is unknown, the
bucket stays `unknown`; the residualizer does not borrow held-out target
popularity. It records `api_called=false`,
`model_training=false`, `server_executed=false`, and
`is_experiment_result=false`; it must not be interpreted as a learned
deconfounding method or paper evidence.

Rerank raw, calibrated, or residualized feature rows through the same
CURE/TRUCE scoring contract:

```powershell
python scripts/rerank_confidence_features.py --features-jsonl outputs/confidence_residuals/<source-run>/popularity_residualized_features.jsonl --confidence-source calibrated_residualized --group-key input_id --top-k 1
```

This writes ignored `reranked_features.jsonl` and `manifest.json` under
`outputs/confidence_reranking/...` by default. The command groups rows by the
requested key, records which confidence source was selected or used as a
fallback, recomputes risk/echo/information-gain components, and emits
`cure_truce_rerank` records with `api_called=false`, `model_training=false`,
`server_executed=false`, and `is_experiment_result=false`. It is a deterministic
integration contract, not a trained reranker and not paper evidence.

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
resume behavior, baseline ranking-to-title adapter behavior, and CURE/TRUCE
exposure-confidence feature-building/scoring/reranking/calibration/popularity
residualization plus calibrated/residualized JSONL reranking scaffold behavior.
Tests use small committed fixtures only and do not download data or call APIs.

## Basic Checks

Governance checks:

```powershell
Get-Location
git branch --show-current
git remote -v
git status --short --branch
```
