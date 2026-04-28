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
  -> analysis summary/reliability/risk cases/run registry
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
wrong-high-confidence cases, correct-low-confidence cases, grounding failures,
parse failures, and a local ignored run registry. These artifacts are required
for reproducibility in later pilots/full runs, but analysis of mock or dry-run
outputs remains a sanity check only.

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

Training is server-oriented unless explicitly approved for a small local
sanity check. Codex must not claim server training or inference has run unless
the user provides logs or result files.

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
decision and requires `--execute-api`, a confirmed provider endpoint/model
config, and the relevant environment variable.

On 2026-04-28, user-approved DeepSeek smoke and 20-example MovieLens sanity
pilot runs were executed with `deepseek-v4-flash`, 10 requests/minute, and max
concurrency 1. Early attempts exposed two engineering issues: local TLS CA
verification needed `certifi`, and the model returned reasoning-only output
until `thinking.type=disabled` was set. After those fixes, the 5-example smoke
and 20-example pilot produced parsed and grounded predictions plus analysis
artifacts. These are small pilot artifacts only, not full-run results or paper
evidence.

Before the first DeepSeek smoke test, run:

```powershell
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --sample-size 5 --stage smoke --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 10 --approved-budget-label USER_APPROVED_SMOKE --execute-api-intended
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
