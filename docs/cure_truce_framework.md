# CURE/TRUCE Framework Scaffold

This document records the first Phase 4 implementation scaffold for the
Storyflow / TRUCE-Rec framework. It is a code contract and test target, not a
trained model result or paper claim.

## Unified Object

The framework is organized around exposure-counterfactual confidence:

```text
C(u, i) ~= P(user accepts item i | user u, do(exposure=1))
```

This object is different from ordinary offline correctness confidence. It asks
whether the user would accept item `i` if the recommender actually exposed it.
That is why the feature schema keeps preference evidence, verbal confidence,
generation evidence, title grounding uncertainty, popularity pressure, novelty,
and observation labels separate.

## Implementation Lineage

This is a storyflow-era scaffold document. The active TRUCE-Rec package is now
`src/llm4rec/`, especially `src/llm4rec/methods/`,
`src/llm4rec/analysis/`, `src/llm4rec/evaluation/`, and
`src/llm4rec/external_baselines/`. Historical `src/storyflow/` paths below are
kept to document the original CURE/TRUCE object, not to supersede active
`llm4rec` code.

## Implemented Code

The scaffold lives in:

```text
src/storyflow/confidence/exposure.py
src/storyflow/confidence/features.py
src/storyflow/confidence/diagnostics.py
src/storyflow/confidence/calibration.py
src/storyflow/confidence/residuals.py
src/storyflow/confidence/reranking.py
src/storyflow/simulation/exposure.py
src/storyflow/triage/reasons.py
scripts/build_confidence_features.py
scripts/calibrate_confidence_features.py
scripts/residualize_confidence_features.py
scripts/rerank_confidence_features.py
scripts/simulate_echo_exposure.py
scripts/triage_confidence_features.py
```

Implemented objects:

- `ExposureConfidenceFeatures`: typed candidate features after title grounding.
- `CureTruceWeights`: deterministic scaffold weights for controlled tests.
- `CureTruceScore`: score, estimated exposure confidence, risk, echo risk,
  information gain, popularity residual, action, and score components.
- `HistogramCalibrator`: split-audited fixed-width calibration scaffold fit
  only on declared fit splits.
- `PopularityResidualModel`: split-audited popularity-bucket mean baseline fit
  only on declared fit splits.
- `SelectedRerankConfidence`: records the selected confidence proxy, available
  sources, fallback status, and missing-source status for reranking.
- `selective_risk_diagnostics`: compact AURC/selective-risk diagnostic over
  selected confidence sources, with head/mid/tail bucket slices.
- `ExposureSimulationConfig`: deterministic synthetic exposure-policy
  configuration over existing feature rows.
- `TriageConfig`: diagnostic reason-code thresholds for uncertainty-aware data
  triage.

Implemented functions:

- `estimate_exposure_confidence`: dynamic evidence average for the proxy
  `C(u, i)`.
- `compute_popularity_residual_confidence`: confidence minus popularity prior.
- `compute_echo_risk`: confidence x popularity x low-novelty pressure.
- `compute_information_gain`: grounded novel tail-candidate exploration value.
- `compute_risk_penalty`: grounding, ambiguity, and overclaim risk.
- `score_cure_truce_candidate`: deterministic CURE/TRUCE score.
- `rerank_cure_truce`: deterministic reranking by score with stable tie break.
- `build_confidence_features`: convert existing grounded observation JSONL
  into CURE/TRUCE feature records plus a manifest.
- `calibrate_feature_rows`: fit and apply a histogram calibration scaffold
  with explicit fit/eval split provenance and leakage guards.
- `residualize_feature_rows`: fit and apply a popularity residual scaffold
  with explicit fit/eval split provenance and leakage guards.
- `rerank_confidence_features_jsonl`: group feature rows, choose raw,
  calibrated, residualized, or combined confidence proxies, recompute
  CURE/TRUCE risk/echo/information-gain components, and write reranked JSONL
  plus a manifest with compact selective-risk diagnostics for the retained
  rows.
- `simulate_exposure_feedback_jsonl`: run synthetic utility/confidence/CURE
  exposure policies over feature rows and write exposure records, metrics, and
  a manifest.
- `triage_features_jsonl`: assign diagnostic reason codes and suggested
  weights for likely-noise candidates, hard tail positives, grounding
  uncertainty, and popularity/echo overconfidence; its manifest includes
  compact selective-risk diagnostics over the selected confidence source.

These functions use no API, no model loading, no training, and no data
download.

## Current Artifact Sanity

On 2026-05-01, the scaffold was exercised on the six existing DeepSeek
Health_and_Personal_Care and Video_Games prompt-gate artifacts:

- Health gate60: free-form, retrieval-context, catalog-constrained;
- Video_Games gate30: free-form, retrieval-context, catalog-constrained.

For each gate, the pipeline wrote ignored local artifacts under:

```text
outputs/confidence_features/api_observations/deepseek/...
outputs/confidence_calibration/api_observations/deepseek/...
outputs/confidence_residuals/api_observations/deepseek/...
outputs/confidence_reranking/api_observations/deepseek/...
outputs/confidence_triage/api_observations/deepseek/...
outputs/echo_simulation/api_observations/deepseek/...
```

The feature counts match the source gate sizes: 60 rows for each Health gate
and 30 rows for each Video_Games gate. Grounded feature counts were Health
free-form 7, Health retrieval-context 53, Health catalog-constrained 32,
Video_Games free-form 9, Video_Games retrieval-context 27, and Video_Games
catalog-constrained 23.

These gates contain only `test` split rows. Calibration and popularity
residualization therefore used `--fit-splits test --eval-splits test
--allow-same-split-eval` only to verify the JSONL contracts. The manifests mark
the fit/eval overlap and `api_called=false`, `model_training=false`,
`server_executed=false`, and `is_experiment_result=false`. This is a leakage
diagnostic for plumbing, not a learned calibrator, not a popularity
deconfounding result, not a trained CURE/TRUCE reranker, and not paper
evidence.

## Feature Builder

Build feature records from a completed grounded observation output:

```powershell
python scripts/build_confidence_features.py --grounded-jsonl outputs/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_api_full185_20260429/grounded_predictions.jsonl --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl --catalog-csv data/processed/amazon_reviews_2023_beauty/full/item_catalog.csv
```

Default output:

```text
outputs/confidence_features/<source-run>/features.jsonl
outputs/confidence_features/<source-run>/manifest.json
```

The builder joins generated-item popularity only when it has the grounded item
catalog row. If no catalog is supplied and a prediction is wrong, it marks
generated popularity as unknown instead of reusing target popularity. This
guard prevents a target-leak artifact from entering later popularity residual
or echo-risk calibration.

## Calibration Scaffold

Run the calibration scaffold only on feature JSONL files that contain proper
split provenance:

```powershell
python scripts/calibrate_confidence_features.py --features-jsonl outputs/confidence_features/<source-run>/features.jsonl --fit-splits train --eval-splits validation,test --n-bins 10
```

Default output:

```text
outputs/confidence_calibration/<source-run>/calibrated_features.jsonl
outputs/confidence_calibration/<source-run>/manifest.json
```

The current scaffold fits a fixed-width empirical mapping from the selected
probability source, defaulting to `score.estimated_exposure_confidence`, to
`feature.correctness_label`. This is the first calibration target for
correctness-labeled observation rows, not the final
exposure-counterfactual-utility target. The manifest records:

- fit splits and evaluation splits;
- split counts in the input feature file;
- fit/eval overlap guard status;
- probability source and label source;
- source and calibrated ECE/Brier summaries on evaluation splits;
- `api_called=false`, `model_training=false`, `server_executed=false`, and
  `is_experiment_result=false`.

By default the command refuses any overlap between fit and evaluation splits.
`--allow-same-split-eval` exists only for explicitly labeled diagnostics and
must not be used for method claims.

## Popularity Residual Scaffold

Run the popularity residual scaffold only on feature JSONL files that contain
proper split provenance:

```powershell
python scripts/residualize_confidence_features.py --features-jsonl outputs/confidence_features/<source-run>/features.jsonl --fit-splits train --eval-splits validation,test
```

Default output:

```text
outputs/confidence_residuals/<source-run>/popularity_residualized_features.jsonl
outputs/confidence_residuals/<source-run>/manifest.json
```

The current scaffold fits a popularity-bucket mean baseline from the selected
probability source, defaulting to `score.estimated_exposure_confidence`, on the
declared fit split only. Evaluation rows receive a
`popularity_residualization` record with:

- source probability;
- popularity bucket and percentile when available;
- popularity-only baseline probability;
- `popularity_residual_confidence`, defined as source confidence minus the
  popularity-only baseline;
- recentered `deconfounded_confidence_proxy`;
- fit/eval split provenance and fallback status.

If generated-item popularity is unknown, the residualizer keeps the bucket as
`unknown` and may fall back to the fit-split global mean. It must not borrow
`target_popularity_bucket` for wrong generated items.

The manifest records split counts, bucket counts, fit/eval overlap guard
status, residual summaries by head/mid/tail bucket, and
`api_called=false`, `model_training=false`, `server_executed=false`, and
`is_experiment_result=false`.

This scaffold is a deconfounding contract for later learned CURE/TRUCE modules.
It is not a learned popularity correction, not a Qwen3 result, and not paper
evidence.

## Reranker Contract

Run the deterministic reranker on raw, calibrated, or residualized feature rows:

```powershell
python scripts/rerank_confidence_features.py --features-jsonl outputs/confidence_residuals/<source-run>/popularity_residualized_features.jsonl --confidence-source calibrated_residualized --group-key input_id --top-k 1
```

Default output:

```text
outputs/confidence_reranking/<source-run>/reranked_features.jsonl
outputs/confidence_reranking/<source-run>/manifest.json
```

The reranker supports these confidence sources:

- `score`: `score.estimated_exposure_confidence`;
- `calibrated`: `calibration.calibrated_probability`, with optional fallback;
- `residualized`: `popularity_residualization.deconfounded_confidence_proxy`,
  with optional fallback;
- `calibrated_residualized`: average of calibrated and residualized proxies
  when both are present, otherwise fallback unless strict mode is requested.

Each output row receives a `cure_truce_rerank` record with the selected
confidence source, fallback reason, group id, rank, score, action, components,
and false API/training/server/result flags. The manifest records input/output
row counts, group counts, split counts, selected-source counts, fallback
counts, compact selective-risk/AURC diagnostics overall and by popularity
bucket, and the row contract. This module is an integration scaffold for later
learned rerankers, not a trained CURE/TRUCE reranker and not paper evidence.

The generated feature rows include:

- `feature`: serialized `ExposureConfidenceFeatures`;
- `score`: deterministic CURE/TRUCE scaffold score;
- `metadata`: source run, target fields, popularity-source provenance, and
  feature-source notes;
- `is_experiment_result=false`.

## Feature Families

The current feature schema separates:

- preference evidence: base utility or ranker score proxy;
- verbal confidence: LLM self-reported probability, treated as noisy;
- generation confidence: token/logprob/sampling proxy when available;
- grounding confidence: title-to-catalog reliability;
- grounding ambiguity: title-to-item uncertainty;
- popularity pressure: percentile or head/mid/tail prior;
- history alignment and novelty: exposure/echo context;
- correctness label: observation or training label, not a default
  inference-time score input;
- grounded status: ungrounded generated titles abstain before correctness
  claims.

This separation is deliberate. Later calibrators can learn which signals
predict exposure-counterfactual acceptance instead of treating all confidence
sources as interchangeable.

## Scoring Contract

The scaffold score combines:

```text
score =
  preference_score_weight * preference_score
  + exposure_confidence_weight * estimated_C
  + information_gain_weight * information_gain
  - risk_penalty_weight * risk_penalty
  - echo_penalty_weight * echo_risk
```

Actions are heuristic and meant for tests:

- `recommend`: grounded, moderate risk, sufficient estimated confidence;
- `diversify`: grounded but echo risk is high;
- `explore`: lower confidence but enough information gain;
- `abstain`: ungrounded or high-risk confidence use.

The score must not be described as a learned CURE/TRUCE method until it is
calibrated and evaluated on approved observation artifacts.

## Echo Simulation And Triage Contracts

The Phase 5 scaffold consumes the same feature rows as calibration,
residualization, and reranking:

```powershell
python scripts/simulate_echo_exposure.py --features-jsonl outputs/confidence_residuals/<source-run>/popularity_residualized_features.jsonl --policies utility_only,confidence_only,utility_confidence,cure_truce --rounds 3 --confidence-source calibrated_residualized
python scripts/triage_confidence_features.py --features-jsonl outputs/confidence_residuals/<source-run>/popularity_residualized_features.jsonl --confidence-source calibrated_residualized
```

The exposure simulation writes `exposure_records.jsonl`,
`simulation_summary.json`, and `manifest.json` under
`outputs/echo_simulation/...`. It reports Exposure Gini, head/mid/tail exposure
share, entropy, synthetic feedback proxy mean, and confidence drift. The
feedback update is synthetic and diagnostic only.

The triage command writes `triaged_features.jsonl` and `manifest.json` under
`outputs/confidence_triage/...`. It adds reason codes and suggested weights
without deleting data. In particular, underconfident correct tail rows are
tagged as hard positives to keep, not pruned as high-uncertainty noise. Its
manifest also records compact selective-risk/AURC diagnostics for the chosen
confidence source and the head/mid/tail slices, without writing the full curve
into the triage manifest.

Both manifests record `api_called=false`, `model_training=false`,
`server_executed=false`, and `is_experiment_result=false`.

## Tests

Run the dedicated scaffold tests:

```powershell
python -m pytest tests/test_confidence_framework.py
python -m pytest tests/test_confidence_feature_builder.py
python -m pytest tests/test_confidence_calibration.py
python -m pytest tests/test_confidence_residuals.py
python -m pytest tests/test_confidence_reranking.py
python -m pytest tests/test_echo_simulation_triage.py
```

The tests cover:

- feature clipping and bucket normalization;
- rejection of invalid labels and buckets;
- ungrounded abstention;
- head/popular echo-risk behavior;
- popularity residual sign;
- reranking toward safer novel tail candidates when echo penalty is high;
- deterministic `top_k` behavior.
- grounded JSONL to feature JSONL conversion;
- generated-item catalog popularity join;
- target-popularity leak guard when catalog data is unavailable;
- CLI manifest writing without API keys.
- fit-split-only histogram calibration;
- fit/eval overlap refusal;
- calibration output/manifest writing without API keys or model training.
- split-fit-only popularity residualization;
- popularity bucket fallback to fit-split global mean;
- residual output/manifest writing without API keys or model training.
- calibrated/residualized confidence-source selection;
- reranking grouped JSONL rows with stable ranks and `top_k`;
- fallback recording and strict missing-source errors;
- rerank output/manifest writing without API keys or model training.
- compact selective-risk diagnostics in rerank and triage manifests.
- synthetic exposure policies, Exposure Gini/tail-share summaries, and
  confidence drift without API calls or model training.
- triage reason codes that preserve hard tail positives and downweight
  wrong-high-confidence popularity/echo cases.

## Next Steps

Short-term framework work should remain API-free:

1. Extend residualization beyond bucket means once approved observation
   evidence supports richer deconfounding.
2. Extend calibration targets from correctness labels toward exposure-
   counterfactual utility once approved exposure/relevance evidence exists.
3. Extend synthetic echo simulation toward approved exposure/relevance
   evidence only after clear manifests exist.
4. Extend reranker integration from deterministic JSONL scoring toward learned
   calibration/reranking once approved observation artifacts and utility
   labels exist.
5. Only after approved server artifacts exist, connect Qwen3-8B + LoRA
   training objectives to this feature schema.

No current file in this scaffold is an experimental result.
