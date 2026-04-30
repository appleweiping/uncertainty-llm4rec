# Storyflow / TRUCE-Rec Experiment Protocol

This protocol defines the intended experimental design. It is not a record of
completed experiments.

## Core Task

The primary task is generative recommendation by item title.

Input:

- A user interaction history represented as item titles and available metadata.
- Optional context such as category, brand, description, timestamp, and
  popularity features when the dataset provides them.

Output:

- One or more generated item titles.
- A confidence statement or probability that the generated recommendation is
  correct.

Mandatory post-processing:

- Every generated title must be grounded to a catalog item before evaluation.
- Ungrounded, ambiguous, duplicated, and out-of-catalog generations must be
  recorded explicitly.

## Grounding Requirement

Generated title grounding maps a free-form string to a catalog item:

```text
generated_title -> normalized candidates -> catalog item or ungrounded
```

The first implementation should prefer transparent matching:

- exact title match after normalization;
- normalized fuzzy match;
- candidate ranking with score and ambiguity metadata.

Later phases may add embedding retrieval, LLM judging, and metadata
disambiguation, but those steps must be logged and evaluated separately.

Grounding outputs must include:

- grounded item id or null;
- grounding method;
- top candidate score;
- second candidate score when available;
- ambiguity indicator;
- match status such as exact, normalized, fuzzy, semantic, ambiguous, or
  out-of-catalog.

## Correctness Targets

The minimal correctness label is next-item correctness:

```text
correct = grounded_item in held_out_next_items
```

The protocol should support both:

- leave-last-one next-item evaluation;
- future-window evaluation with multiple held-out future interactions.

Future phases may add graded relevance when ratings, categories, or semantic
audits support it. The binary implicit-feedback label must not be described as
the only possible user preference truth.

## Preprocessing And Split Policy

The data pipeline supports two recommendation evaluation settings.

### Per-User Leave-Last Setting

Each user's interactions are sorted chronologically. The held-out target is
defined from the end of that user's sequence:

- leave-last-one: the final item is the test target;
- leave-last-two: the second-to-last item is the validation target and the
  final item is the test target;
- earlier eligible items can produce training examples when they have enough
  history.

This setting is useful for next-item recommendation where every evaluated user
has a personal historical context.

### Rolling Examples With Global Chronological Split

For each user sequence, the pipeline can create iterative examples where target
index `k` ranges over `[min_history, n - 1]`. The history is the prefix before
`k`, optionally truncated to `max_history`. These rolling examples are then
sorted by target timestamp and split globally into train/validation/test.

This setting is useful when the experiment needs a global time boundary and
multiple examples per user, matching iterative sequential recommendation
setups.

### Filtering Policy

The pipeline supports user interaction-count filtering and iterative k-core
filtering over users and items. These filters are included to follow common
recommendation preprocessing settings and to avoid evaluating users/items with
too little behavioral evidence for sequential history construction.

TODO: add exact citations after related-work reading.

## Confidence Analysis

Confidence must always be analyzed jointly with:

- correctness;
- item popularity;
- grounding confidence and ambiguity;
- head/mid/tail group;
- user history length and user mainstreamness when available;
- category or semantic group when available.

Core confidence outcomes:

- correct and confident;
- correct but uncertain;
- wrong but uncertain;
- wrong and confident.

Planned confidence metrics:

- ECE and adaptive ECE where implemented;
- Brier score;
- CBU_tau: correct but uncertain;
- WBC_tau: wrong but confident;
- AURC/selective risk where implemented;
- reliability diagram data split by popularity bucket;
- popularity-confidence slope after controlling for correctness and grounding;
- tail underconfidence gap.

## Observation Schedule

Observation must start small and real before scaling:

1. Validate schemas and grounding on fixtures.
2. Run a small real-data pilot after Phase 1 data processing exists.
3. Inspect pilot logs for parse failures, grounding errors, confidence format
   errors, cache behavior, and cost/rate-limit risk.
4. Only then run larger real-data observations with manifests and resumable
   JSONL output.

No full-run claim may be written before the corresponding full run exists.

## Phase 2A Generative Observation File Flow

The practical observation pipeline is:

```text
processed examples
  -> observation inputs
  -> raw responses
  -> parsed predictions
  -> grounded predictions
  -> metrics/report/manifest
  -> analysis summary/reliability/repeat slices/risk cases/run registry
```

Processed examples come from `data/processed/<dataset>/<run>/` and must contain
or be joinable to history titles, target title, timestamp, item popularity, and
head/mid/tail bucket. Observation inputs are JSONL records with prompt text and
prompt hash. Raw responses are kept separate from parsed predictions. Grounded
predictions must record generated title, grounded item id, grounding status,
grounding score, ambiguity, correctness, confidence, popularity bucket, and
provider metadata.

Phase 2A uses `provider=mock` only. Mock outputs validate parsing, grounding,
metrics, and resume behavior without calling any external API. Mock metrics are
sanity checks, not API pilot results and not paper evidence.

Phase 2C analysis reads grounded predictions, failed cases, and run manifests
to produce reliability diagram bins, popularity-bucket summaries,
repeat-target summaries, wrong-high-confidence cases, correct-low-confidence
cases, grounding failures, parse failures, and a local ignored run registry.
Repeat-target summaries are required for e-commerce data where repeat purchases
or duplicate review artifacts can put the held-out target in the user history.
These artifacts are required for reproducibility in later pilots/full runs, but
analysis of mock or dry-run outputs remains a sanity check only.

## Baseline Observation Contract

Baselines must enter the project through the same title-level path:

```text
baseline selected/ranked item
  -> catalog title
  -> grounded prediction
  -> correctness + confidence proxy + popularity + grounding analysis
```

Lightweight popularity and train-split co-occurrence baselines are local sanity
layers. The `ranking_jsonl` adapter is the contract for later SASRec,
BERT4Rec, GRU4Rec, LightGCN, and similar ranking baselines: they may produce a
ranked item list, but the project evaluates the selected title only after
catalog lookup and grounding. Rank/score-derived confidence values are proxies
unless a later calibration stage validates them. No ranking baseline output
should bypass title grounding or be reported as a paper result without a full
protocol run and manifest.

External ranking artifacts must pass two validation-manifest gates before they
are adapted into grounded title predictions. First, the upstream run manifest
gate checks baseline family/model/run labels, train/evaluation split
separation, command/git/seed provenance, declared input and ranking paths, and
grounding/leakage guard flags. Second, the artifact gate checks coverage over
the selected observation input slice, duplicate records, supported ranking
JSONL schema, score metadata, catalog item-id compatibility, already-seen
history items, split/dataset declarations, and upstream provenance fields where
available. These gates do not train or run the upstream ranker; they only
record whether the local artifact is safe to enter the title-grounding
observation path.

Once a baseline observation run is analyzed, its analysis summary and local run
registry must preserve `source_kind=baseline_observation` and
`confidence_semantics=non_calibrated_baseline_proxy`. This lets baselines share
the same correctness/popularity/grounding/head-tail analysis schema while
preventing rank/popularity/co-occurrence confidence proxies from being
described as calibrated LLM confidence.

## Framework Stage

The framework stage targets Qwen3-8B + LoRA or a comparable small-model
training setup.

Planned components:

- SFT baseline for title generation and confidence expression;
- RecBrier-style confidence objective where feasible;
- popularity residual or deconfounding features;
- exposure-aware scoring and reranking;
- risk-aware preference optimization where feasible;
- uncertainty-guided data triage that distinguishes noise from hard tail
  positives.

The current committed scaffold starts with the common object:

```text
C(u, i) ~= P(user accepts item i | user u, do(exposure=1))
```

`storyflow.confidence.ExposureConfidenceFeatures` keeps preference evidence,
verbal confidence, generation evidence, grounding confidence/ambiguity,
popularity pressure, history alignment, novelty, and observed correctness
labels separate. The deterministic CURE/TRUCE score is a testable contract for
future calibrators and rerankers; it is not a trained probability, not a Qwen3
result, and not paper evidence. The current histogram calibration scaffold is
also only a split-provenance and leakage-guard contract: it fits on declared
fit splits, evaluates on declared evaluation splits, and records both in a
manifest. The current popularity residual scaffold is likewise only a
split-provenance and leakage-guard contract: it fits a popularity-bucket mean
confidence baseline on declared fit splits, applies it to declared evaluation
splits, and writes residualized feature rows plus a manifest.

Grounded observation outputs can be converted into the feature schema with
`scripts/build_confidence_features.py`. The feature builder may join the
processed item catalog to attach generated-item popularity. When catalog data
is absent and the grounded item is not the target, generated-item popularity
must remain unknown rather than borrowing target popularity. This protects the
popularity residual and echo-risk analysis from target leakage.

Feature rows with proper train/validation/test provenance can be passed through
`scripts/calibrate_confidence_features.py`. The command defaults to
`fit_splits=train` and `eval_splits=validation,test`, refuses overlap unless an
explicit diagnostic flag is used, and writes ignored calibrated feature rows
plus a manifest. Its first target is `feature.correctness_label`; later
exposure-counterfactual utility targets require approved exposure/relevance
evidence and must not be simulated into method claims.

The same feature rows can be passed through
`scripts/residualize_confidence_features.py`. The command defaults to fitting
on train and applying to validation/test, refuses fit/eval overlap unless an
explicit diagnostic flag is used, and writes ignored
`popularity_residualized_features.jsonl` plus a manifest under
`outputs/confidence_residuals/`. The residual is source confidence minus the
fit-split popularity-bucket mean baseline, with a recentered
`deconfounded_confidence_proxy` for later framework modules. This is not a
learned deconfounding method, not a full-result analysis, and not paper
evidence.

Raw, calibrated, or residualized feature rows can then be passed through
`scripts/rerank_confidence_features.py`. The command defaults to
`confidence_source=calibrated_residualized`, groups rows by `input_id`, writes
ignored `reranked_features.jsonl` plus a manifest under
`outputs/confidence_reranking/`, and records the selected confidence source,
fallback status, score components, action, rank, and false API/training/server
flags. This is a deterministic reranker contract only. It is not a trained
CURE/TRUCE reranker, not a Qwen3 result, and not paper evidence.

The same feature rows can now enter a Phase 5 scaffold:

```text
CURE/TRUCE feature rows
  -> synthetic exposure simulation
  -> diagnostic data triage
```

`scripts/simulate_echo_exposure.py` evaluates API-free synthetic policies:
`utility_only`, `confidence_only`, `utility_confidence`, and `cure_truce`.
The feedback signal is explicitly synthetic: it uses an existing correctness
label when available, otherwise the preference-score proxy. The command
reports Exposure Gini, head/mid/tail exposure share, entropy, and confidence
drift, but these diagnostics must not be described as real feedback or method
evidence.

`scripts/triage_confidence_features.py` writes diagnostic reason codes and
suggested weights. It separates likely-noise candidates, grounding uncertainty,
popularity/echo overconfidence, and hard tail positives. It must not be used as
a final pruning policy without later approved training/evaluation evidence,
and it explicitly preserves underconfident hard tail positives.

Training is server-oriented unless explicitly approved for a small local
sanity check. Codex must not claim server training or inference has run unless
the user provides logs or result files.

## Qwen3 Server Observation Interface

Qwen3-8B observation is a Phase 3 server interface, not a completed run. The
committed scaffold is:

- `configs/server/qwen3_8b_observation.yaml`;
- `scripts/server/run_qwen3_observation.py`;
- `storyflow.server`.

By default, the script is plan-only and writes a server job manifest plus an
API-compatible output contract. It must record `api_called=false`,
`server_executed=false`, `model_inference_run=false`, and
`is_experiment_result=false`.

Actual server inference requires `--execute-server` and must write the same
layers as API observation:

```text
request_records
  -> raw_responses
  -> parsed_predictions / failed_cases
  -> grounded_predictions
  -> metrics/report/manifest
  -> analysis summary/risk cases/run registry
```

Every Qwen3 generated title must be grounded to the catalog before correctness
or confidence metrics are computed. No local Codex run may be described as a
server or Qwen3 result without user-provided logs and artifacts.

## Local Versus Server Split

Local tasks:

- repository editing and documentation;
- small fixtures and unit tests;
- MovieLens-scale preprocessing where feasible;
- pilot API observation after explicit configuration;
- metric and plotting code from completed outputs;
- Chinese local reports.

Server tasks:

- Qwen3-8B full inference if too heavy locally;
- Qwen3-8B + LoRA training;
- large Amazon category preprocessing and training;
- large baseline runs;
- long-running observation or simulation.

Server scripts must write manifests, logs, config snapshots, seed values, output
paths, git commit hash, and environment summaries.

## Data Policy

Planned dataset support:

- MovieLens 1M for local real-data sanity checks.
- Amazon Reviews 2023 categories for full-scale generative recommendation.
- Steam/games data if a reliable source is selected.
- Additional Amazon categories as configs allow.

Raw data belongs under `data/raw/` and must not be committed. If a source
requires login, license acceptance, or manual placement, the pipeline must fail
clearly and produce a Chinese report with the expected path and resume command.

MovieLens 1M success is not sufficient for the final experimental story. After
local MovieLens validation, the project must move to at least one Amazon
Reviews 2023 full-data category, starting with Beauty unless a later protocol
change gives a stronger reason to choose another category first. The project
must not remain indefinitely in small-data sanity mode.

Local raw files are currently available for Amazon Reviews 2023 Beauty,
Digital_Music, Handmade_Products, and Health_and_Personal_Care. Beauty remains
the first full-data gate; the additional categories are robustness candidates
after Beauty sample/full preparation is reproducible.

Amazon Beauty local sample preparation must be labeled as readiness only. The
sample command uses `--sample-mode --max-records N`, writes
`is_sample_result=true` and `is_experiment_result=false` to the preprocess
manifest, and is not paper evidence. Full Amazon preparation requires explicit
approval and `--allow-full`.

## API Policy

The first API pilot target is DeepSeek only. API adapters for Qwen/DashScope,
Kimi/Moonshot, and GLM/Zhipu remain future phases. API keys must come from
environment variables and must never be committed.

API runs must support:

- provider and model selection;
- rate limits and concurrency;
- retries with exponential backoff;
- response caching and idempotent resume;
- JSONL inputs and outputs;
- cost/token accounting when available;
- separation of prompts, requests, raw responses, parsed predictions, and
  grounded predictions.

Paid APIs must not be called in tests.

Phase 2B adds provider configs and a dry-run API runner. Dry-run mode is the
default and must not read API keys or call the network. Real API execution is
allowed only after an explicit user-approved provider/model/budget/rate-limit
/concurrency decision and requires `--execute-api`, a confirmed provider
endpoint/model config, and the relevant environment variable.

On 2026-04-28, user-approved DeepSeek smoke and 20-example MovieLens sanity
pilot runs were executed with `deepseek-v4-flash`, 10 requests/minute, and max
concurrency 1. Early attempts exposed two engineering issues: local TLS CA
verification needed `certifi`, and the model returned reasoning-only output
until `thinking.type=disabled` was set. After those fixes, the 5-example smoke
and 20-example pilot produced parsed and grounded predictions plus analysis
artifacts. These are small pilot artifacts only, not full-run results or paper
evidence.

On the same date, after Amazon Beauty sample readiness, a user-approved
30-example Amazon Beauty sample pilot was executed with `deepseek-v4-flash`, 30
requests/minute, max concurrency 3, cache enabled, and
`execution_mode=execute_api`. An earlier diagnostic attempt exposed that dry-run
and real API cache entries must be separated; the runner now includes
`execution_mode` in the cache key and rejects cached responses whose `dry_run`
flag does not match the current run. The corrected 30-example pilot is a sample
pilot only, not a full Amazon/API run or paper result.

Before the first DeepSeek smoke test, run:

```powershell
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --sample-size 5 --stage smoke --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 10 --approved-max-concurrency 1 --approved-budget-label USER_APPROVED_SMOKE --execute-api-intended
```

This command is a gate check only and does not call the API.

## Reporting Rules

Reports, papers, and README updates may say:

- implemented;
- planned;
- not yet run;
- synthetic demo only;
- pilot;
- full run.

They must not say that a method improves performance unless the claim is backed
by actual code, logs, metrics, and output paths.
