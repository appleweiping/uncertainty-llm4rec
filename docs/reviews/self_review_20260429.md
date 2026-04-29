# Self Review 2026-04-29

This self-review follows the project rule to audit the research direction after
several substantial commits. It is a governance artifact, not an experimental
result.

## Current Phase

Storyflow / TRUCE-Rec is between Phase 2C and Phase 3:

- Phase 0 governance/scaffold is complete.
- Phase 1 real-data preprocessing exists for MovieLens and Amazon Reviews 2023
  local/sample/full entry points.
- Phase 2 mock, API dry-run, DeepSeek pilot, observation analysis, case review,
  grounding diagnostics, and lightweight baseline observation interface exist.
- Phase 3 full observation is not complete because full Amazon prepared data,
  larger API/full observation, Qwen3/server observation, and heavier baselines
  are not yet executed.

## Mainline Check

The project still matches the core Storyflow task:

```text
user history item titles
  -> generated item title
  -> catalog grounding
  -> correctness + confidence + popularity + grounding analysis
```

The current code does not collapse into ranking-only recommendation. Even the
new baseline layer emits a title and writes the same grounded prediction schema
as the API/mock runners.

## Grounding Check

All generated or selected titles in the current observation layers must be
grounded before correctness:

- mock observation grounds `generated_title`;
- API observation grounds parsed `generated_title`;
- baseline observation grounds selected baseline title;
- case review and grounding diagnostics operate on grounded predictions.

Remaining risk: the free-form DeepSeek Amazon sample pilot had grounding
failures in earlier diagnostics. This is a prompt/catalog QA risk, not a reason
to remove grounding. The retrieval-context and catalog-constrained gates exist
to diagnose it before scale-up.

## Confidence Analysis Check

The project analyzes confidence jointly with:

- correctness;
- GroundHit and grounding ambiguity;
- head/mid/tail popularity bucket;
- wrong-high-confidence cases;
- correct-low-confidence cases;
- Tail Underconfidence Gap;
- exploratory popularity-confidence slope.

Remaining risk: current observed numbers are pilot/sample or baseline sanity
artifacts. They must not be written as paper evidence until full runs exist.

## Toy-Risk Check

Risk status: yellow.

The codebase has moved beyond synthetic fixtures and MovieLens:

- local raw Amazon files exist for Beauty, Digital_Music, Handmade_Products,
  and Health_and_Personal_Care;
- Amazon Beauty sample prepare, validation, observation input gates, DeepSeek
  sample pilot, case review, and baseline observation sanity exist;
- additional local Amazon category configs and lightweight inspect gates exist.

The remaining toy-risk is staying too long at `sample_5k` and 30-example
pilots. The next priority should be a concrete Amazon Beauty full-prepare or
sample-to-full gate, not more small diagnostic layers unless they unblock full
scale-up.

## Dataset Route

Current route is coherent:

1. Synthetic fixtures: tests only.
2. MovieLens 1M: local sanity and low-cost pilot substrate only.
3. Amazon Beauty: first full e-commerce category.
4. Digital_Music / Handmade / Health: local raw robustness categories after
   Beauty is reproducible.
5. Sports / Video_Games / Books: later robustness/server categories.
6. Steam/Yelp: optional, source/license gated.

Immediate gap: `docs/dataset_matrix.md` still marks Beauty next action as
validation/sample before full prepare. That should be updated after a real full
prepare gate or full prepare run exists.

## Baseline Route

Current baseline support is a correct first layer, not sufficient final
coverage:

- implemented: popularity and train-split co-occurrence title baselines;
- missing: ranking-to-title adapter for SASRec, BERT4Rec, GRU4Rec, LightGCN;
- missing: generative recommendation baselines such as P5-like, TIGER-style, or
  BIGRec-like references where feasible.

Next baseline step should not be heavy training. The next safe step is a
ranking-to-title output adapter and config contract so future trained baselines
write the same observation schema.

## Qwen3 / Server Route

The server runbook separates local work from server work and does not claim any
server result. Qwen3-8B and Qwen3-8B + LoRA remain planned.

Immediate gap: there is not yet a `scripts/server/` observation scaffold that
defines the exact JSONL input/output contract for Qwen3-8B inference. This is a
high-priority non-training task after the Amazon full-data gate.

## Framework Coherence

The framework story is still conceptually coherent but not yet implemented.
The unifying object remains:

```text
C(u, i) ~= P(user accepts item i | user u, do(exposure=1))
```

Planned modules should serve this object:

- verbal confidence: noisy observation;
- token/logprob/sampling evidence: generation evidence;
- grounding confidence: title-to-item uncertainty;
- popularity residual: confounding correction;
- exposure-aware score: echo-risk control;
- triage: distinguish noise from hard-tail-positive evidence.

Risk: if Phase 4 adds calibration, debiasing, reranking, and triage as separate
tricks, the project will look stitched together. Implementation must start from
the exposure-counterfactual confidence object and map each module to it.

## Reviewer-Attack Risks

1. Single-provider evidence risk.
   Current real API pilots are DeepSeek only. Mitigation: keep provider
   abstraction and add Qwen/server observation scaffold before paper claims.

2. Small-sample evidence risk.
   Current Amazon API evidence is sample-level. Mitigation: move to Beauty
   full-data prepare and full observation gates.

3. Grounding-failure confounding.
   Confidence may reflect grounding ease instead of preference. Mitigation:
   report GroundHit, ambiguity, failure taxonomy, and separate grounded vs
   ungrounded analysis.

4. Popularity baseline insufficiency.
   Lightweight baselines are not final reviewer-proof baselines. Mitigation:
   add ranking-to-title adapter and later server-trained sequential baselines.

5. Implicit-feedback target ambiguity.
   Next item is not the only possible user utility truth. Mitigation: support
   future-window and graded relevance when data allows.

6. Sample_5k stagnation.
   Repeated pilots on sample data can look toy-like. Mitigation: prioritize
   full Beauty prepare and validation.

7. Framework stitching risk.
   CURE/TRUCE must not be a list of unrelated uncertainty tricks. Mitigation:
   use exposure-counterfactual confidence as the common object.

8. Server-result claim risk.
   Qwen3/server paths are documented but not run. Mitigation: continue explicit
   `not yet run` wording until logs/artifacts exist.

9. API cost/cache contamination risk.
   Earlier dry-run/API cache separation was fixed. Mitigation: keep
   `execution_mode` in cache keys and maintain manifest gates.

10. Data leakage risk.
    Candidate-constrained prompts can leak target items if configured
    incorrectly. Mitigation: keep `allow_target_in_candidates=false` by default
    and mark constrained prompts as diagnostic, not accuracy evidence.

## Priority Fixes

P0:

- Keep generated outputs, raw data, processed data, local reports, reviewer
  report, and API caches untracked.
- Do not claim pilot/full/server results without manifests and logs.

P1:

- Run or prepare a concrete Amazon Beauty full-data gate.
- Validate full processed output before any full API observation.

P2:

- Add Qwen3/server observation scaffold with the same JSONL schema and no
  claim of execution.

P3:

- Add ranking-to-title baseline adapter contract.

P4:

- Start CURE/TRUCE framework schema from exposure-counterfactual confidence,
  not from disconnected calibration/debias/triage components.

## Decision

Go forward. The mainline remains valid, but the next work should reduce
sample/pilot dependence. The strongest next non-server task is Amazon Beauty
full prepare gate and validation; the strongest next server-oriented scaffold
task is Qwen3-8B observation I/O without running inference.
