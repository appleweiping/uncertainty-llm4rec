# Ablation Protocol

This protocol defines Phase 6/7 ablations for Calibrated
Uncertainty-Guided Generative Recommendation. It is a plan for comparable real
experiments and smoke validation, not a result table.

## Shared requirements

- All variants emit the unified prediction JSONL schema.
- All variants use the same shared evaluator.
- Comparable variants use the same split and candidate protocol.
- Train-only statistics are used for popularity and novelty.
- Target title, target item ID, future interactions, and correctness labels are
  forbidden as prompt or policy inputs.
- MockLLM outputs are smoke evidence only.

## Ours variants

### Ours full

- Config path: `configs/methods/ours_uncertainty_guided.yaml`.
- Experiment template: `configs/experiments/real_ours_method_template.yaml`.
- Allowed inputs: history, non-target candidates, catalog, train popularity,
  generated title, confidence, grounding score, candidate-normalized
  confidence, history similarity, same-candidate fallback scores.
- Disabled components: none.
- Output schema: unified prediction schema with `ours_method=true` and
  `ablation_variant=full`.
- Metrics: ranking, validity, hallucination, confidence, calibration, coverage,
  diversity, novelty, long-tail, latency/cost.
- Leakage risks: target in prompt, test popularity, fallback candidate mismatch.

### Ours fallback-only

- Config path: `configs/methods/ours_fallback_only.yaml`.
- Experiment template: `configs/experiments/real_ablation_template.yaml`.
- Allowed inputs: history, train data for fallback, same candidate set.
- Disabled components: generation acceptance, uncertainty policy, grounding
  acceptance, candidate-normalized confidence, popularity adjustment, echo-risk
  guard.
- Output schema: unified prediction schema with `ablation_variant=fallback_only`.
- Metrics: all shared metrics.
- Leakage risks: fallback using non-comparable candidates or non-train evidence.

### Ours w/o uncertainty

- Config path: `configs/methods/ours_ablation_no_uncertainty.yaml`.
- Disabled components: uncertainty policy.
- Expected output: `disabled_components=["uncertainty_policy"]`.
- Leakage risks: hidden confidence threshold still affecting decisions.

### Ours w/o grounding

- Config path: `configs/methods/ours_ablation_no_grounding.yaml`.
- Disabled components: grounding check as accept criterion.
- Expected output: `disabled_components=["grounding_check"]`.
- Leakage risks: mapping ungrounded generations using target labels.

### Ours w/o candidate-normalized confidence

- Config path:
  `configs/methods/ours_ablation_no_candidate_normalized_confidence.yaml`.
- Disabled components: candidate-normalized confidence prompt and threshold.
- Expected output: `disabled_components=["candidate_normalized_confidence"]`.
- Leakage risks: computing normalized confidence under another metadata key.

### Ours w/o popularity adjustment

- Config path: `configs/methods/ours_ablation_no_popularity_adjustment.yaml`.
- Disabled components: popularity overconfidence adjustment.
- Expected output: `disabled_components=["popularity_adjustment"]`.
- Leakage risks: popularity flag still changing policy decisions.

### Ours w/o echo guard

- Config path: `configs/methods/ours_ablation_no_echo_guard.yaml`.
- Disabled components: history-similarity / echo-risk guard.
- Expected output: `disabled_components=["echo_risk_guard"]`.
- Leakage risks: hidden fallback on high history similarity.

## Baseline comparators

### LLM generative baseline

- Config path: `configs/experiments/smoke_llm_generative.yaml` for smoke;
  real template: `configs/experiments/real_llm_api_template.yaml`.
- Disabled components: Ours uncertainty policy, fallback routing, popularity
  adjustment, echo-risk guard.
- Leakage risks: target in prompt.

### LLM rerank baseline

- Config path: `configs/experiments/smoke_llm_rerank.yaml`.
- Disabled components: free-form title generation and Ours policy.
- Leakage risks: reranker seeing target title under an incompatible protocol.

### BM25

- Config path: `configs/experiments/smoke_bm25.yaml`.
- Disabled components: LLM calls and Ours policy.
- Leakage risks: query text includes target title not present in history.

### Popularity

- Config path: `configs/experiments/smoke_popularity.yaml`.
- Disabled components: LLM calls, grounding policy, text scoring.
- Leakage risks: popularity computed from validation/test.

### MF

- Config path: `configs/experiments/smoke_mf.yaml`.
- Disabled components: text, LLM, Ours policy.
- Leakage risks: fitting on held-out interactions.

### Sequential Markov / sequential baseline

- Config path: `configs/experiments/smoke_sequential.yaml` or
  `configs/experiments/smoke_phase4_all.yaml`.
- Disabled components: LLM and Ours policy.
- Leakage risks: transitions fit from held-out future examples.

## Ablation tables

Real ablation tables must be filled only from completed `metrics.json` files.
Columns should include method, disabled component, candidate protocol, seeds,
ranking metrics, validity metrics, calibration metrics, long-tail metrics,
latency/cost, and artifact paths. TBD: fill from real metrics files.
