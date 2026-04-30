# Self Review 2026-04-30

This self-review follows the Qwen3 server observation scaffold, the
ranking-JSONL baseline adapter, the Phase 4 CURE/TRUCE framework scaffold, the
first Phase 5 echo simulation / data-triage scaffold, and the Amazon
cross-category readiness matrix. It is a governance and research-quality
artifact, not an experimental result.

## Current Phase

Storyflow / TRUCE-Rec is now between Phase 3, early Phase 4, and scaffolded
Phase 5.

- Phase 0 governance/scaffold is complete.
- Phase 1 data support is beyond MovieLens: Amazon Beauty local full
  preprocessing exists as an ignored data-readiness artifact, and several
  additional Amazon raw categories are available locally for later robustness.
- Phase 2 mock/API dry-run, DeepSeek observation, observation analysis, case
  review, candidate diagnostics, run comparison, grounding diagnostics, and
  run registry are implemented.
- Phase 3 is partially open: Qwen3-8B server observation has a plan/execution
  contract, and baseline observation now supports popularity, co-occurrence,
  and ranking-JSONL-to-title conversion. The baseline route now also has a
  source run-manifest validator, a ranking artifact validator, and
  baseline-aware source semantics in the analysis registry. No Qwen3
  inference, server run, LoRA training, heavy baseline training, or
  paper-result experiment has been run by Codex.
- Phase 4 is partially open as API-free scaffold work: feature schema,
  grounded-observation feature builder, deterministic CURE/TRUCE scoring,
  split-audited histogram calibration, and split-audited popularity
  residualization are implemented and tested. The calibrated/residualized JSONL
  reranker contract is also implemented and tested. None of these are learned
  method results or paper evidence.
- Phase 5 has an API-free scaffold: synthetic exposure simulation and
  diagnostic data triage consume CURE/TRUCE feature rows and write ignored
  non-result manifests.
- Cross-category data readiness now includes a matrix over configured Amazon
  Reviews 2023 categories, including server-scale Books, Video_Games, and
  Sports_and_Outdoors entries. This is readiness only, not a full data run.

## External Reviewer Blocker Status

The ignored `docs/reviewer_report.md` from 2026-04-29 flagged an unclosed
baseline observation change as a blocker. The first blocker was resolved by
commit `bc4deec feat: add ranking baseline adapter`, which:

- keeps baseline outputs title-level;
- grounds selected titles before correctness/confidence/popularity analysis;
- adds a ranking-JSONL adapter contract for future SASRec/BERT4Rec/GRU4Rec/
  LightGCN-style rankers;
- runs and passes the dedicated baseline tests and the full pytest suite.

The follow-up reviewer-proofing gap is now also closed at the contract level:

- `500564b feat: add baseline artifact validation gate` validates coverage,
  schema, catalog compatibility, history overlap, split declarations, and
  upstream provenance before a ranking artifact enters `ranking_jsonl`.
- `31be990 feat: add baseline run manifest validation` validates the upstream
  baseline source run manifest, including run identity, split separation,
  command/git/seed provenance, declared artifact paths, and leakage guard
  flags.
- `c309775 feat: add baseline-aware analysis registry` preserves
  `source_kind`, `claim_scope`, `confidence_semantics`, and claim guardrails so
  baseline confidence proxies are not confused with calibrated LLM confidence.

The reviewer report itself remains ignored and is not committed.

## Mainline Check

The project still matches the Storyflow task:

```text
user history item titles
  -> generated or selected item title
  -> catalog grounding
  -> correctness + confidence + popularity + grounding + head/mid/tail analysis
```

The project has not collapsed into ordinary top-k recommendation. Even ranking
baselines are required to pass through catalog-title lookup and title grounding
before entering analysis. This is the right guardrail for reviewer-proof
baseline expansion.

## Grounding Check

All current observation paths require grounding before correctness:

- mock observation grounds the generated title;
- API observation grounds parsed generated titles;
- baseline observation grounds selected titles from popularity, co-occurrence,
  or ranking JSONL;
- Qwen3 server observation scaffold defines API-compatible generated-title
  layers that must be grounded before metrics;
- analysis and comparison layers read grounded prediction artifacts.

The main remaining risk is not a missing grounding step; it is that free-form
Amazon Beauty title generation still has substantial grounding difficulty. The
existing retrieval-context and catalog-constrained runs are diagnostic prompt
and grounding gates, not recommendation-accuracy evidence, because they exclude
the held-out target from candidates by design.

## Confidence Analysis Check

The project currently analyzes confidence jointly with:

- correctness;
- GroundHit and grounding status;
- target and generated head/mid/tail buckets where available;
- wrong-high-confidence and correct-low-confidence cases;
- Tail Underconfidence Gap when correct head/tail rows exist;
- exploratory popularity-confidence slope;
- repeat-target slices for e-commerce data;
- candidate-set adherence, selected rank, target-leak status, and
  history-copying for diagnostic candidate prompts.

For baseline runs, confidence values are explicitly proxies. The ranking-JSONL
adapter records score/rank metadata for schema compatibility, but it must not
be described as calibrated model confidence unless a later calibration stage
validates it.

## Phase 4 Scaffold Check

The Phase 4 scaffold now has concrete code around the unified target:

```text
C(u, i) ~= P(user accepts item i | user u, do(exposure=1))
```

Implemented pieces:

- `ExposureConfidenceFeatures` keeps preference evidence, verbal confidence,
  generation evidence, grounding confidence/ambiguity, popularity pressure,
  history alignment, novelty, correctness label, and grounded status separate.
- `build_confidence_features` converts grounded observation JSONL into
  feature JSONL, with generated-item popularity provenance and a guard against
  borrowing target popularity for wrong predictions.
- `calibrate_feature_rows` fits a split-audited histogram scaffold on declared
  fit splits and applies it to declared evaluation splits only.
- `residualize_feature_rows` fits a split-audited popularity-bucket confidence
  baseline on declared fit splits and applies it to evaluation splits only.
  Unknown generated-item popularity remains `unknown`; the residualizer does
  not borrow held-out target buckets.
- `rerank_confidence_features_jsonl` consumes raw, calibrated, or residualized
  feature rows; records selected confidence source and fallback provenance; and
  writes rank/action/risk/echo/information-gain components plus a manifest.
  It is deterministic integration code, not a trained reranker.

This is coherent with Storyflow because each module serves the same object:
verbal confidence is treated as noisy evidence; grounding confidence captures
title-to-item uncertainty; popularity residualization is a confounding
correction; reranking uses the selected confidence proxy to control exposure
risk. The main risk is now less "no framework implementation" and more "do not
overclaim deterministic scaffolds as learned CURE/TRUCE method evidence."

## Toy-Risk Check

Risk status: yellow, improving.

The project is no longer stuck at synthetic fixtures or MovieLens. Amazon
Beauty full local preprocessing exists, and a scoped DeepSeek full185
repeat-free diagnostic family exists under ignored outputs. However, the
project can still look too narrow if it keeps extending only DeepSeek prompt
diagnostics. The next high-value work should broaden model-family and method
coverage without making unsupported claims.

Immediate mitigation:

- keep Amazon Beauty as the first full category but add a server/cross-category
  readiness route for Video_Games or Books after the protocol stabilizes;
- use the Qwen3 server scaffold for approved server observation rather than
  making local Qwen claims;
- move the next API-free framework work toward baseline provenance,
  exposure-simulation, or triage contracts that consume the same grounded
  feature/rerank schema, instead of adding another independent trick.

## Dataset Route

Current route is coherent:

1. Synthetic fixtures: tests only.
2. MovieLens 1M: local sanity and low-cost API substrate only.
3. Amazon Beauty: first full e-commerce category and active full-slice
   diagnostic substrate.
4. Digital_Music, Handmade, and Health: local raw robustness candidates after
   Beauty.
5. Video_Games and Sports: larger title-rich/e-commerce robustness categories.
6. Books: long-tail title-rich server-scale category.
7. Steam/Yelp: optional and source/license gated.

Next dataset work should not return to MovieLens. The cross-category readiness
path is now reproducible; the next dataset step should be to use it to decide
which title-rich category, likely Video_Games or Books, gets raw placement and
server/manual full-preparation approval after Beauty.

## Baseline Route

Current support is now a stronger contract layer, not sufficient final
reviewer coverage.

- Implemented: popularity title baseline.
- Implemented: train-split co-occurrence title baseline.
- Implemented: ranking-JSONL-to-title adapter contract.
- Implemented: source run-manifest validation for future trained rankers.
- Implemented: ranking artifact validation before title-grounded adaptation.
- Implemented: analysis registry semantics that mark baseline confidence as a
  non-calibrated proxy.
- Needed next: one approved trained baseline artifact, produced outside this
  local Codex task or on server, must pass both validation gates before it is
  adapted into grounded title observations.
- Needed later: trained SASRec/BERT4Rec/GRU4Rec/LightGCN runs, followed by
  P5-like, TIGER/Semantic-ID-like, BIGRec/grounding-style, and uncertainty-
  aware generative baselines where reproducible.

The correct next baseline step is no longer another local manifest scaffold.
It is to use the contract on an actual trained ranking artifact after user
approval, while keeping local work focused on API-free modules that consume the
same grounded observation/feature schema.

## Qwen3 / Server Route

No server experiment, Qwen3-8B inference, or Qwen3-8B + LoRA training has been
run by Codex. The server route is now more concrete than the prior review:

- `configs/server/qwen3_8b_observation.yaml` defines the observation contract;
- `scripts/server/run_qwen3_observation.py` writes plan-only artifacts by
  default;
- `storyflow.server` mirrors API observation layers for future server runs;
- tests cover plan-only and schema behavior.

Remaining gap: server execution still needs explicit user approval, a real
server environment, model source, input slice, artifact-return policy, logs,
and manifests. Until then, all Qwen3 wording must remain scaffold/planned.

## Framework Coherence

The unifying object remains:

```text
C(u, i) ~= P(user accepts item i | user u, do(exposure=1))
```

Current CURE/TRUCE implementation makes this object explicit in code, but only
as a scaffold:

- verbal confidence: noisy observation of exposure-counterfactual confidence;
- token/logprob/sampling: generation evidence;
- grounding confidence: title-to-item uncertainty;
- popularity residual: confounding correction already implemented as a
  split-audited bucket-mean scaffold;
- exposure-aware score: echo-risk control implemented as deterministic
  contract, and the JSONL reranker now consumes calibrated/residualized
  confidence proxies with manifest provenance;
- triage: separate likely noise from hard-tail-positive evidence.

The next framework step should not add another independent trick. Echo
simulation/data triage now consume the same grounded feature/rerank records.
Any learned calibrator/reranker or utility target must wait for approved
observation artifacts and explicit server/training gates.

## Reviewer-Attack Risks

1. Single-provider evidence.
   Priority: high. DeepSeek diagnostics are useful but not generality evidence.
   Mitigation: keep Qwen API/server Qwen3 routes explicit; do not claim
   cross-model behavior before approved runs exist.

2. Candidate-diagnostic correctness confusion.
   Priority: high. Retrieval-context and catalog-constrained prompts exclude
   the target. Mitigation: continue to label them prompt/candidate/grounding QA
   and never report their target correctness as recommendation accuracy.

3. Grounding-failure confounding.
   Priority: high. Free-form confidence can reflect catalog grounding ease.
   Mitigation: report GroundHit, grounding status, ambiguity, candidate
   adherence, and ungrounded-high-confidence cases with all confidence metrics.

4. Framework stitching and scaffold overclaim.
   Priority: high. Calibration, popularity residualization, reranking, and
   triage could still look disconnected or be mistaken for a learned method.
   Mitigation: every next module must consume the existing grounded
   feature/rerank contracts, keep false API/training/server/result flags, and
   remain tied to `C(u, i) ~= P(user accepts | do(exposure=1))`.

5. Baseline insufficiency.
   Priority: high. Lightweight baselines are not final reviewer-proof coverage.
   Mitigation: the manifest and artifact gates now exist; next, an approved
   SASRec/BERT4Rec/GRU4Rec/LightGCN artifact must pass both gates and then flow
   through ranking-to-title grounding before analysis.

6. Small-slice overinterpretation.
   Priority: medium-high. Full185 is a scoped diagnostic family, not a full
   suite. Mitigation: broaden to multi-provider, server, and cross-category
   only with manifests and approval.

7. Implicit-feedback ambiguity.
   Priority: medium-high. Next item is not the only preference truth.
   Mitigation: support future-window and graded relevance paths where data
   allows, and state binary next-item limitations in reports.

8. Popularity as relevance versus confounding.
   Priority: medium. Reviewers may object that popularity is valid signal.
   Mitigation: keep residualization split-audited, report it as a popularity-
   only confidence baseline, and avoid treating popularity residuals as proof
   of preference deconfounding before learned/evaluated evidence exists.

9. Server claim risk.
   Priority: medium. Server scripts exist but have not run. Mitigation: keep
   every Qwen/server statement plan-only until logs/artifacts are inspected.

10. Long-term data breadth risk.
    Priority: medium. Beauty alone is too narrow. Mitigation: after the Beauty
    protocol stabilizes, move to Video_Games/Books or another title-rich Amazon
    category.

## Priority Fixes

P0:

- Keep raw responses, API cache, raw/processed data, outputs, runs, local
  reports, reviewer report, PDFs, and zips untracked.
- Keep all pilot/diagnostic/full-slice wording scoped and non-paper-result.

P1:

- Use the baseline source-manifest and artifact gates on the first approved
  trained ranking artifact; do not train or claim one locally without approval.
- Add echo-simulation or data-triage scaffolds only if they consume existing
  grounded feature/rerank records and stay explicitly synthetic/scaffold.

P2:

- Use the cross-category Amazon readiness matrix to plan raw placement and
  server/manual preparation for a title-rich category such as Video_Games or
  Books.
- Prepare a user-approval checklist for the next real API/provider expansion
  without running it.

P3:

- Add echo simulation and data triage only after the feature schema and
  observation contracts are stable enough to avoid a stitched design.

## Decision

Go forward. The previous baseline blocker is closed, the project remains
title-level and grounding-first, and Phase 4 is now partially implemented as
tested scaffolding. The calibrated/residualized reranker contract is now also
closed by `f857eac feat: add CURE TRUCE reranker contract`. The baseline
contract layer is also stronger after `500564b`, `31be990`, and `c309775`.
The next non-blocking engineering work should be to prepare approval gates for
the next real expansion: either a trained ranking artifact entering the
validated baseline path, a Qwen3/server observation run, or a title-rich Amazon
category raw-placement/full-prepare step. Do not run any of these without
explicit approval, manifests, and artifact-return rules.

Do not start Qwen3 inference, LoRA training, server execution, or another real
API expansion without explicit user approval and concrete run gates.
