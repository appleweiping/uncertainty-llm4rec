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

## Implemented Code

The scaffold lives in:

```text
src/storyflow/confidence/exposure.py
src/storyflow/confidence/features.py
src/storyflow/confidence/calibration.py
scripts/build_confidence_features.py
scripts/calibrate_confidence_features.py
```

Implemented objects:

- `ExposureConfidenceFeatures`: typed candidate features after title grounding.
- `CureTruceWeights`: deterministic scaffold weights for controlled tests.
- `CureTruceScore`: score, estimated exposure confidence, risk, echo risk,
  information gain, popularity residual, action, and score components.
- `HistogramCalibrator`: split-audited fixed-width calibration scaffold fit
  only on declared fit splits.

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

These functions use no API, no model loading, no training, and no data
download.

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

## Tests

Run the dedicated scaffold tests:

```powershell
python -m pytest tests/test_confidence_framework.py
python -m pytest tests/test_confidence_feature_builder.py
python -m pytest tests/test_confidence_calibration.py
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

## Next Steps

Short-term framework work should remain API-free:

1. Add a learned or fit-on-observation popularity residual module.
2. Extend calibration targets from correctness labels toward exposure-
   counterfactual utility once approved exposure/relevance evidence exists.
3. Add reranker integration that can consume API, Qwen3, and baseline
   grounded outputs.
4. Only after approved server artifacts exist, connect Qwen3-8B + LoRA
   training objectives to this feature schema.

No current file in this scaffold is an experimental result.
