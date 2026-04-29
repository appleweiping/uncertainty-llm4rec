# Self Review 2026-04-29

This self-review updates the earlier 2026-04-29 phase gate after the
repeat-free Amazon Beauty full185 DeepSeek diagnostic sequence and the
observation-run comparison layer. It is a governance artifact, not an
experimental result.

## Current Phase

Storyflow / TRUCE-Rec is between Phase 2C and Phase 3.

- Phase 0 governance/scaffold is complete.
- Phase 1 data support has moved beyond MovieLens: Amazon Beauty local full
  prepare was executed as a data-readiness artifact, and additional local raw
  Amazon categories are available for later robustness.
- Phase 2 mock/API dry-run/DeepSeek API observation, analysis, case review,
  candidate diagnostics, comparison, grounding diagnostics, and lightweight
  baseline interfaces exist.
- Phase 3 is only partially open: one DeepSeek Amazon Beauty repeat-free
  full-slice diagnostic family exists, but multi-provider observation,
  Qwen3/server observation, heavy baselines, framework training, simulation,
  and paper artifacts have not run.

## Mainline Check

The project still matches the Storyflow task:

```text
user history item titles
  -> generated/selected item title
  -> catalog grounding
  -> correctness + confidence + popularity + grounding + candidate analysis
```

The code has not collapsed into a ranking-only recommender. Even popularity
and co-occurrence baselines emit titles and pass through catalog grounding
before metrics.

## Grounding Check

All current observation layers require grounding before correctness:

- mock observation grounds `generated_title`;
- API observation grounds parsed `generated_title`;
- baseline observation grounds selected baseline title;
- analysis, case review, candidate diagnostics, and comparison read grounded
  prediction artifacts.

Current diagnostic evidence remains scoped:

- free-form DeepSeek full185 has low GroundHit on Amazon Beauty and therefore
  should not be scaled blindly;
- retrieval-context improves prompt/candidate grounding behavior on the same
  slice;
- catalog-constrained is stricter but leaves more ungrounded low-confidence
  cases;
- both candidate-context diagnostics exclude the held-out target, so target-hit
  correctness is not recommendation accuracy.

## Confidence Analysis Check

The project currently analyzes confidence jointly with:

- correctness;
- GroundHit and grounding status;
- head/mid/tail target and generated buckets where available;
- wrong-high-confidence and correct-low-confidence cases;
- Tail Underconfidence Gap when correct head/tail rows exist;
- exploratory popularity-confidence slope;
- candidate-set adherence, selected rank, target-leak status, and
  history-copying for candidate-context prompts.

The newly added comparison layer correctly writes guardrails: it is prompt,
candidate, and grounding QA, not a paper-result table and not a method
improvement claim.

## Toy-Risk Check

Risk status: yellow, improving.

The project is no longer stuck at MovieLens or synthetic fixtures. Amazon
Beauty full local preprocessing and DeepSeek full185 repeat-free diagnostics
exist. However, the project can still look toy-like if it keeps adding small
diagnostics without moving to cross-provider, server, baseline, and framework
coverage.

Immediate mitigation:

- use the full185 prompt comparison to choose the next observation strategy;
- add Qwen3/server observation scaffolding before claiming model-family
  generality;
- keep Amazon Beauty as the first full category, then extend to at least one
  title-rich robustness category such as Video_Games or Books when the protocol
  is stable.

## Dataset Route

Current route is coherent:

1. Synthetic fixtures: tests only.
2. MovieLens 1M: local sanity and low-cost API substrate only.
3. Amazon Beauty: first full e-commerce category and current active full-slice
   diagnostic substrate.
4. Digital_Music, Handmade, and Health: local raw robustness candidates after
   Beauty.
5. Sports and Video_Games: larger robustness categories.
6. Books: long-tail title-rich server-scale category.
7. Steam/Yelp: optional, source/license gated.

Next dataset task should be either a reproducible cross-category sample gate
or a server-ready full-category runbook, not another MovieLens milestone.

## Baseline Route

Current support is a correct first layer, not sufficient final reviewer
coverage.

- Implemented: popularity and train-split co-occurrence title baselines.
- Needed next: ranking-to-title adapter contract for SASRec, BERT4Rec, GRU4Rec,
  LightGCN, and similar sequential/ranking baselines.
- Needed later: P5-like, TIGER/Semantic-ID-like, BIGRec/grounding-style, and
  uncertainty-aware baselines if reproducible.

The next baseline step should stay API-free and training-free: define the
output schema and conversion contract so future trained baselines enter the
same grounding/confidence/analysis pipeline.

## Qwen3 / Server Route

No server experiment, Qwen3-8B inference, or Qwen3-8B + LoRA training has been
run by Codex. The server runbook is honest but still mostly a placeholder.

Priority gap:

- add `scripts/server/` and `configs/server/` scaffolds for Qwen3-8B
  observation I/O;
- make the output schema match API observation: raw/parsed/grounded/failed,
  manifest, cache/resume, and analysis compatibility;
- do not run inference or training locally; do not claim server results.

## Framework Coherence

The unifying object remains:

```text
C(u, i) ~= P(user accepts item i | user u, do(exposure=1))
```

Future CURE/TRUCE implementation must map each module to this object:

- verbal confidence: noisy confidence observation;
- token/logprob/sampling: generation evidence;
- grounding confidence: title-to-item uncertainty;
- popularity residual: confounding correction;
- exposure-aware score: echo-risk control;
- triage: separate noise from hard-tail-positive evidence.

The project must avoid adding calibration, debiasing, reranking, and triage as
unconnected tricks. Each future implementation and ablation should point back
to exposure-counterfactual confidence.

## Reviewer-Attack Risks

1. Single-provider evidence.
   DeepSeek diagnostics are useful but insufficient. Fix priority: add Qwen API
   or Qwen3/server observation path before any paper claim of generality.

2. Candidate-diagnostic correctness confusion.
   Retrieval/context and catalog-constrained prompts exclude the target. Fix
   priority: keep guardrails in comparison reports and never report their
   target-hit correctness as recommendation accuracy.

3. Grounding-failure confounding.
   Free-form confidence may reflect catalog grounding ease. Fix priority:
   report GroundHit, grounding status, ambiguity, and candidate adherence with
   all confidence metrics.

4. Small-slice overinterpretation.
   Full185 is a useful local full-slice diagnostic, not a full experimental
   suite. Fix priority: add cross-provider and cross-category gates.

5. Baseline insufficiency.
   Lightweight baselines are not final reviewer-proof baselines. Fix priority:
   implement ranking-to-title adapter contracts before heavy training.

6. Implicit-feedback ambiguity.
   Next item is not the only preference truth. Fix priority: support
   future-window and graded relevance paths where data allows.

7. Popularity as relevance versus confounding.
   Reviewers may argue popularity is valid signal. Fix priority: model
   popularity-supported preference separately from popularity-only confidence.

8. Framework stitching.
   CURE/TRUCE could look like a bag of tricks. Fix priority: implement feature
   schema and scoring around `P(user accepts | do(exposure))`.

9. Server claim risk.
   Qwen/server paths are not run. Fix priority: keep all server language as
   planned/scaffold until logs/artifacts exist.

10. Long-term data breadth risk.
    Beauty alone is not enough. Fix priority: after Beauty protocol is stable,
    move to Video_Games/Books or another title-rich Amazon category.

## Priority Fixes

P0:

- Keep raw responses, API cache, data, outputs, local reports, reviewer report,
  PDFs, and zips untracked.
- Keep all pilot/diagnostic/full-slice wording scoped and non-paper-result.

P1:

- Add Qwen3/server observation scaffold with compatible JSONL schema and no
  inference claim.
- Add ranking-to-title baseline adapter contract.

P2:

- Decide the next DeepSeek/Qwen API gate from the prompt comparison, not from a
  vague desire to scale.
- Add at least one cross-category Amazon readiness route after Beauty.

P3:

- Start the CURE/TRUCE feature schema and scoring scaffold from
  exposure-counterfactual confidence.

P4:

- Add echo simulation and data triage only after the observation and framework
  objects are stable enough to avoid stitched-together design.

## Decision

Go forward with caution. The mainline remains strong and non-toy, but the next
work should reduce single-provider and prompt-diagnostic dependence. The
strongest non-blocking next tasks are:

1. Qwen3-8B/server observation scaffold without execution;
2. ranking-to-title baseline adapter contract;
3. CURE/TRUCE feature schema from exposure-counterfactual confidence.
