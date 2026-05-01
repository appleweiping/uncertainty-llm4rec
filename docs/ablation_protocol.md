# Phase 6 Ablation Protocol

This protocol defines smoke-ready variants for Calibrated
Uncertainty-Guided Generative Recommendation. All variants must output the
shared prediction schema and be evaluated by the shared evaluator.

## Shared evaluation setup

- Dataset config: `configs/datasets/tiny.yaml` for Phase 6 smoke tests.
- Split: `test`.
- Candidate protocol: full tiny candidate set from preprocessing.
- LLM provider: MockLLM only.
- Seed: `13`.
- Evaluator: `llm4rec.evaluation.evaluator.evaluate_predictions`.
- Metrics: ranking, validity/hallucination, confidence, calibration, coverage,
  diversity, novelty, long-tail, latency/token efficiency.

## Variants

### Ours full

- Config path: `configs/methods/ours_uncertainty_guided.yaml`.
- Allowed inputs: history, non-target visible candidates, catalog, train
  popularity, MockLLM output, grounding result, candidate-normalized confidence,
  fallback ranker scores over the same candidate set.
- Disabled components: none.
- Expected output schema: unified prediction JSONL with `ours_method=true` and
  `ablation_variant=full`.
- Evaluation metrics: all shared metrics.
- Leakage risks: target title or ID appearing in prompt; test popularity used by
  policy; fallback using a different candidate set.

### Ours w/o uncertainty

- Config path: `configs/methods/ours_ablation_no_uncertainty.yaml`.
- Allowed inputs: history, non-target visible candidates, catalog, MockLLM
  generated title, grounding result, fallback ranker if grounding fails.
- Disabled components: uncertainty policy thresholds and confidence-based
  fallback.
- Expected output schema: unified prediction JSONL with
  `disabled_components=["uncertainty_policy"]`.
- Evaluation metrics: all shared metrics plus decision metadata counts.
- Leakage risks: accidentally retaining confidence threshold decisions.

### Ours w/o grounding

- Config path: `configs/methods/ours_ablation_no_grounding.yaml`.
- Allowed inputs: history, non-target visible candidates, MockLLM generated
  title/confidence, fallback ranker.
- Disabled components: grounding check as a required accept criterion.
- Expected output schema: unified prediction JSONL with
  `disabled_components=["grounding_check"]`.
- Evaluation metrics: all shared metrics, especially validity and hallucination.
- Leakage risks: using target labels to map an ungrounded title.

### Ours w/o candidate-normalized confidence

- Config path: `configs/methods/ours_uncertainty_guided.yaml` with
  `method.params.ablation.variant=no_candidate_normalized_confidence`, or the
  grouped ablation experiment config.
- Allowed inputs: history, non-target visible candidates, catalog, MockLLM
  generation, grounding, popularity, history similarity.
- Disabled components: candidate-normalized confidence prompt and threshold.
- Expected output schema: unified prediction JSONL with
  `disabled_components=["candidate_normalized_confidence"]`.
- Evaluation metrics: all shared metrics plus confidence/calibration.
- Leakage risks: still computing normalized confidence under another metadata
  name.

### Ours w/o popularity adjustment

- Config path: `configs/methods/ours_ablation_no_popularity_adjustment.yaml`.
- Allowed inputs: history, non-target visible candidates, catalog, MockLLM
  generation, grounding, candidate-normalized confidence, history similarity.
- Disabled components: popularity overconfidence adjustment in policy.
- Expected output schema: unified prediction JSONL with
  `disabled_components=["popularity_adjustment"]`.
- Evaluation metrics: all shared metrics plus popularity-stratified metrics.
- Leakage risks: using popularity risk flag to affect decisions despite the
  ablation.

### Ours w/o history-similarity / echo-risk guard

- Config path: `configs/methods/ours_ablation_no_echo_guard.yaml`.
- Allowed inputs: history, non-target visible candidates, catalog, MockLLM
  generation, grounding, candidate-normalized confidence, popularity.
- Disabled components: echo-risk guard decision rule.
- Expected output schema: unified prediction JSONL with
  `disabled_components=["echo_risk_guard"]`.
- Evaluation metrics: all shared metrics plus diversity, novelty, and history
  similarity metadata summaries.
- Leakage risks: retaining hidden fallback on high history similarity.

### Ours fallback-only

- Config path: `configs/methods/ours_fallback_only.yaml`.
- Allowed inputs: history, candidate set, train data used by fallback ranker.
- Disabled components: generation acceptance, uncertainty policy acceptance,
  grounding acceptance, candidate-normalized confidence, popularity adjustment,
  echo-risk guard.
- Expected output schema: unified prediction JSONL with
  `ablation_variant=fallback_only` and fallback metadata.
- Evaluation metrics: all shared metrics.
- Leakage risks: fallback must use the same candidate set and train-only data.

### LLM generative baseline

- Config path: `configs/experiments/smoke_llm_generative.yaml`.
- Allowed inputs: Phase 3 generative prompt inputs and MockLLM output.
- Disabled components: Ours policy, fallback, popularity adjustment, echo guard.
- Expected output schema: unified prediction JSONL with
  `not_ours_method=true`.
- Evaluation metrics: all shared metrics.
- Leakage risks: target leakage in generation prompt.

### LLM rerank baseline

- Config path: `configs/experiments/smoke_llm_rerank.yaml`.
- Allowed inputs: history and visible candidates.
- Disabled components: title generation, Ours policy, fallback routing.
- Expected output schema: unified prediction JSONL with
  `not_ours_method=true`.
- Evaluation metrics: all shared metrics.
- Leakage risks: target candidate removal rules must match documented protocol.

### BM25

- Config path: `configs/experiments/smoke_bm25.yaml`.
- Allowed inputs: history item text, catalog text, candidate set.
- Disabled components: LLM generation, Ours policy, external APIs.
- Expected output schema: unified prediction JSONL.
- Evaluation metrics: all shared metrics.
- Leakage risks: target title cannot be used as query except if already present
  in history.

### Popularity

- Config path: `configs/experiments/smoke_popularity.yaml`.
- Allowed inputs: train examples and candidate set.
- Disabled components: LLM generation, grounding, text scoring.
- Expected output schema: unified prediction JSONL.
- Evaluation metrics: all shared metrics.
- Leakage risks: popularity must come from train split only.

### MF

- Config path: `configs/experiments/smoke_mf.yaml`.
- Allowed inputs: train interactions/examples and candidate set.
- Disabled components: LLM generation, text retrieval, Ours policy.
- Expected output schema: unified prediction JSONL.
- Evaluation metrics: all shared metrics.
- Leakage risks: train-only fitting and same split.

### Sequential Markov / sequential baseline

- Config path: `configs/experiments/smoke_sequential.yaml` or
  `configs/experiments/smoke_phase4_all.yaml`.
- Allowed inputs: train sequences, user history, candidate set.
- Disabled components: LLM generation, grounding, Ours policy.
- Expected output schema: unified prediction JSONL.
- Evaluation metrics: all shared metrics.
- Leakage risks: transition counts must be trained from train examples only.
