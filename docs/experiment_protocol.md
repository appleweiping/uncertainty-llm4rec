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

## API Policy

API adapters are planned for DeepSeek, Qwen/DashScope, Kimi/Moonshot, and
GLM/Zhipu. API keys must come from environment variables and must never be
committed.

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
