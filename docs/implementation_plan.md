# Storyflow / TRUCE-Rec Implementation Plan

This document turns the Storyflow research thesis into a staged engineering
plan. It is intentionally scoped as governance and planning material only. It
does not contain experimental results.

## Guiding Principles

- `Storyflow.md` is the conceptual source of truth.
- The task is title-level generative recommendation, not only top-k ranking.
- A generated title must be grounded to a catalog item before correctness or
  confidence is evaluated.
- Confidence must be analyzed with correctness, popularity, grounding, and
  head/mid/tail group structure.
- Local work and server-only work must be separated in commands, docs, reports,
  and claims.
- No result can be claimed without code, config, logs, manifests, and output
  paths.

## Phase 0: Governance And Scaffold

Goal: establish the repository rules, documentation skeleton, artifact policy,
and execution protocol before any modeling work begins.

Deliverables:

- Verify active directory, branch, and remote.
- Commit `AGENTS.md` and `Storyflow.md` as governing documents if they are not
  already tracked.
- Create `.gitignore` rules for secrets, local reports, raw data, caches,
  outputs, runs, API caches, PDFs, and zip files.
- Update `README.md` with project identity, status markers, milestone map, and
  safety policy.
- Create implementation, experiment, Codex execution, server, and reference
  documentation.
- Create a Chinese local report under `local_reports/`.

Exit criteria:

- Repository is on `main`.
- Remote is `https://github.com/appleweiping/uncertainty-llm4rec.git`.
- No data is downloaded.
- No API is called.
- No model or toy model is implemented.
- Basic repository checks pass.
- Changes are committed and pushed to `origin/main`.

## Phase 1: Data, Download, And Preprocessing

Goal: add real-data pipeline support without drifting into toy-only mode.

Planned deliverables:

- Dataset manifest format with source, license/access notes, checksum when
  available, local raw path, processed path, and command provenance.
- MovieLens 1M downloader and processor for a fast local real-data sanity
  check.
- Amazon Reviews 2023 configs for full-scale categories such as Beauty,
  Sports_and_Outdoors, Toys_and_Games, Video_Games, Books, CDs_and_Vinyl, and
  Office_Products.
- Optional Steam/games data plan if a reliable source is selected.
- K-core and interaction-count filtering.
- Chronological splitting, leave-last-one, leave-last-two validation/test, and
  rolling examples.
- History truncation, title cleaning, metadata joining, popularity computation,
  and head/mid/tail bucket assignment.
- Tests for data loading, filtering, splitting, title normalization, popularity
  buckets, and deterministic fixture generation.

Exit criteria:

- At least MovieLens 1M is processed as a real local sanity-check dataset.
- Restricted datasets fail loudly with a Chinese report and resume command.
- Raw data and caches remain uncommitted.

## Phase 2: API Observation Pipeline

Goal: build the first generative observation pipeline for title generation,
catalog grounding, confidence extraction, and correctness labeling.

Planned deliverables:

- Prompt templates for title-level generative recommendation.
- JSONL input/output schema that separates prompt, request, raw response,
  parsed prediction, grounded prediction, confidence, and labels.
- Provider abstraction with mock provider tests. Paid APIs are not used in
  tests.
- Config-driven provider/model selection for DeepSeek, Qwen/DashScope,
  Kimi/Moonshot, and GLM/Zhipu in later real runs.
- Rate limit, concurrency, retry, cache, resume, and run manifest design.
- Title grounding module using exact/normalized match first, then fuzzy or
  embedding-based candidates in later iterations.
- Correctness labels against next item and future-window targets.
- Confidence extraction from verbal probability or structured output.
- Small pilot observation on real processed data after the pipeline is tested.

Exit criteria:

- Mock-provider tests pass.
- Pilot output is explicitly labeled pilot.
- No paid API is called without explicit user approval and configured keys.

## Phase 3: Full Observation And Baselines

Goal: scale observation across providers, datasets, and baseline families while
keeping provenance complete.

Planned deliverables:

- Batch runner with prompt deduplication, cache keys, idempotent resume, and
  partial-run continuation.
- Multi-provider observation for official large-model APIs.
- Local/server Qwen3-8B observation support when feasible.
- Baseline observation support for sequential recommenders and generative
  LLM4Rec baselines in phased order.
- Metrics for Recall@K, NDCG@K, Hit Ratio@K, MRR@K, coverage, and tail coverage.
- Generative/title metrics: GroundHit, grounding confidence, grounding
  ambiguity, out-of-catalog rate, duplicate title rate, fuzzy/semantic match
  status.
- Confidence metrics: ECE, adaptive ECE if implemented, Brier score, CBU_tau,
  WBC_tau, AURC/selective risk if implemented, and reliability diagram data by
  popularity bucket.
- Popularity-confidence coupling analysis, tail underconfidence, and
  wrong-high-confidence analysis.

Exit criteria:

- All full-run claims are backed by tracked configs, logs, manifests, and output
  paths.
- Observations distinguish synthetic, pilot, and full runs.

## Phase 4: CURE/TRUCE Framework

Goal: implement the core exposure-aware confidence framework.

Planned deliverables:

- Triangulated uncertainty features from verbal confidence, token likelihood,
  sampling consistency, semantic dispersion, grounding confidence, popularity
  residuals, and counterfactual stability where feasible.
- Calibrator for correctness and exposure-counterfactual utility targets.
- Popularity residual/deconfounding module that separates preference-supported
  popularity from popularity-only confidence.
- Exposure-aware scoring and reranking.
- RecBrier-style confidence objective where feasible.
- Risk-aware preference optimization design.
- Qwen3-8B + LoRA training scripts and configs for server execution.
- Server runbook updates with exact commands once scripts exist.

Exit criteria:

- Framework behavior is tested on controlled fixtures before full runs.
- Server-only training is documented but not claimed unless actual logs are
  provided.

## Phase 5: Echo Simulation And Data Triage

Goal: evaluate how confidence affects exposure feedback and use uncertainty for
data triage without deleting long-tail signal.

Planned deliverables:

- Confidence-induced exposure simulation.
- Multi-round feedback loop with deterministic seeds.
- Exposure Gini, tail exposure share, category entropy, and confidence drift.
- Uncertainty-guided triage with decomposition of likely noise, epistemic hard
  positives, aleatoric ambiguity, grounding uncertainty, and popularity-induced
  confidence.
- Noise injection experiments for controlled validation.
- Tests for simulation determinism and triage behavior.

Exit criteria:

- Synthetic noise experiments are labeled synthetic.
- Triage does not use naive high-uncertainty pruning as the final policy without
  decomposition and evaluation.

## Phase 6: Full Experiments And Paper Artifacts

Goal: produce a reproducible full experimental suite and paper-ready artifacts
from actual logs only.

Planned deliverables:

- Full dataset runs.
- Full baseline suite where feasible.
- Full model runs including small-model framework variants.
- Final tables and plots generated from actual output manifests.
- Related-work notes, baseline comparisons, and reproducibility package.
- Paper-ready analysis that distinguishes observation, method, simulation, and
  triage evidence.

Exit criteria:

- Every table and plot links to a concrete run manifest.
- No fabricated result, placeholder result, or undocumented manual edit appears
  in paper artifacts.
