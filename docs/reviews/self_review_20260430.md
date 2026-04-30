# Self Review 2026-04-30

This self-review follows the Qwen3 server observation scaffold and the
ranking-JSONL baseline adapter commits. It is a governance and research-quality
artifact, not an experimental result.

## Current Phase

Storyflow / TRUCE-Rec is still between Phase 2C and Phase 3.

- Phase 0 governance/scaffold is complete.
- Phase 1 data support is beyond MovieLens: Amazon Beauty local full
  preprocessing exists as an ignored data-readiness artifact, and several
  additional Amazon raw categories are available locally for later robustness.
- Phase 2 mock/API dry-run, DeepSeek observation, observation analysis, case
  review, candidate diagnostics, run comparison, grounding diagnostics, and
  run registry are implemented.
- Phase 3 is partially open: Qwen3-8B server observation has a plan/execution
  contract, and baseline observation now supports popularity, co-occurrence,
  and ranking-JSONL-to-title conversion. No Qwen3 inference, server run, LoRA
  training, heavy baseline training, or paper-result experiment has been run by
  Codex.

## External Reviewer Blocker Status

The ignored `docs/reviewer_report.md` from 2026-04-29 flagged an unclosed
baseline observation change as a blocker. That blocker is resolved by commit
`bc4deec feat: add ranking baseline adapter`, which:

- keeps baseline outputs title-level;
- grounds selected titles before correctness/confidence/popularity analysis;
- adds a ranking-JSONL adapter contract for future SASRec/BERT4Rec/GRU4Rec/
  LightGCN-style rankers;
- runs and passes the dedicated baseline tests and the full pytest suite.

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
- connect the next framework code directly to exposure-counterfactual
  confidence.

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

Next dataset work should not return to MovieLens. It should either make an
Amazon cross-category sample/readiness path reproducible or prepare a
server-ready full-category runbook for a title-rich domain.

## Baseline Route

Current support is a stronger first layer, not sufficient final reviewer
coverage.

- Implemented: popularity title baseline.
- Implemented: train-split co-occurrence title baseline.
- Implemented: ranking-JSONL-to-title adapter contract.
- Needed next: baseline artifact manifests that record model family, training
  split, seed, config, and ranked-output provenance.
- Needed later: trained SASRec/BERT4Rec/GRU4Rec/LightGCN runs, followed by
  P5-like, TIGER/Semantic-ID-like, BIGRec/grounding-style, and uncertainty-
  aware generative baselines where reproducible.

The correct next baseline step is still API-free and training-free: standardize
the artifact manifest and split-audit contract before any heavy baseline run.

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

Future CURE/TRUCE implementation must make this object explicit in code, not
only in prose:

- verbal confidence: noisy observation of exposure-counterfactual confidence;
- token/logprob/sampling: generation evidence;
- grounding confidence: title-to-item uncertainty;
- popularity residual: confounding correction;
- exposure-aware score: echo-risk control;
- triage: separate likely noise from hard-tail-positive evidence.

The next framework scaffold should define a typed feature schema and scoring
contract around this object before adding calibrators, rerankers, or triage
rules.

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

4. Framework stitching.
   Priority: high. Calibration, debiasing, reranking, and triage could look
   disconnected. Mitigation: implement feature schema and scoring around
   `P(user accepts | do(exposure))`.

5. Baseline insufficiency.
   Priority: high. Lightweight baselines are not final reviewer-proof coverage.
   Mitigation: add baseline artifact manifests, then run approved heavy
   baselines through the ranking-to-title adapter.

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
   Mitigation: separate preference-supported popularity from popularity-only
   confidence residuals.

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

- Start the CURE/TRUCE feature schema from exposure-counterfactual confidence.
- Add a baseline artifact manifest contract for future trained rankers.

P2:

- Add a cross-category Amazon readiness route after Beauty, prioritizing a
  title-rich domain such as Video_Games or Books when storage/server planning
  is clear.
- Prepare a user-approval checklist for the next real API/provider expansion
  without running it.

P3:

- Add echo simulation and data triage only after the feature schema and
  observation contracts are stable enough to avoid a stitched design.

## Decision

Go forward. The previous baseline blocker is closed, the project remains
title-level and grounding-first, and the next non-blocking engineering work
should begin Phase 4 carefully: define the CURE/TRUCE feature and scoring
schema around exposure-counterfactual confidence before implementing a full
calibrator, reranker, or training objective.
