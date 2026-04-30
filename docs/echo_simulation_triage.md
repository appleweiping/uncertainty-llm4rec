# Echo Simulation And Data Triage Scaffold

This document records the first API-free Phase 5 scaffold for confidence-guided
exposure simulation and uncertainty-aware data triage. It is a contract layer
over existing CURE/TRUCE feature rows. It is not a real feedback experiment,
not model training, not a server run, and not paper evidence.

## Input Contract

Both commands consume CURE/TRUCE feature JSONL rows produced by
`scripts/build_confidence_features.py`, optionally after calibration,
popularity residualization, or reranking:

```text
grounded predictions
  -> CURE/TRUCE feature rows
  -> optional calibration / popularity residualization / reranking
  -> echo simulation or data triage
```

The input rows must preserve a `feature` object built after title grounding.
Ungrounded rows may remain in the file, but they are explicitly marked and must
not be used for correctness claims.

## Echo Simulation

Run the synthetic exposure simulation:

```powershell
python scripts/simulate_echo_exposure.py --features-jsonl outputs/confidence_residuals/<source-run>/popularity_residualized_features.jsonl --policies utility_only,confidence_only,utility_confidence,cure_truce --rounds 3 --confidence-source calibrated_residualized
```

Default outputs are ignored under:

```text
outputs/echo_simulation/<source-run>/exposure_records.jsonl
outputs/echo_simulation/<source-run>/simulation_summary.json
outputs/echo_simulation/<source-run>/manifest.json
```

Implemented policies:

- `utility_only`: exposes the candidate with the highest preference-score
  proxy.
- `confidence_only`: exposes the candidate with the highest selected
  confidence proxy.
- `utility_confidence`: exposes by a weighted average of preference and
  confidence.
- `cure_truce`: uses the CURE/TRUCE deterministic score with risk, echo-risk,
  and information-gain components.

The multi-round update uses synthetic feedback only. If a row has an observed
`correctness_label`, that label is used as a synthetic feedback proxy; otherwise
the feature preference score is used. This is a diagnostic feedback loop for
plumbing and metrics, not a model of real user clicks.

The summary reports:

- Exposure Gini over candidate items;
- head/mid/tail exposure share;
- popularity-bucket entropy;
- category entropy when category metadata is available;
- mean exposed confidence;
- synthetic feedback proxy mean;
- confidence drift across rounds.

All manifests write `synthetic_feedback=true`, `api_called=false`,
`model_training=false`, `server_executed=false`, and
`is_experiment_result=false`.

## Data Triage

Run diagnostic reason-code triage:

```powershell
python scripts/triage_confidence_features.py --features-jsonl outputs/confidence_residuals/<source-run>/popularity_residualized_features.jsonl --confidence-source calibrated_residualized
```

Default outputs are ignored under:

```text
outputs/confidence_triage/<source-run>/triaged_features.jsonl
outputs/confidence_triage/<source-run>/manifest.json
```

The triage layer adds `data_triage` to each row with:

- action: `keep`, `review`, `downweight`, or `prune_candidate`;
- reason codes;
- suggested training weight;
- selected confidence source;
- correctness label when available;
- popularity bucket, grounding, novelty, and echo-risk diagnostics.

The triage manifest also records `selective_risk_diagnostics`: a compact AURC
and selective-risk summary over the requested confidence source, both overall
and by popularity bucket. This connects observation-level selective risk to
the downstream triage contract without treating the triage output as a result
or a final pruning policy.

Important reason codes:

- `hard_tail_positive_underconfident`: keep and upweight underconfident correct
  tail positives.
- `wrong_high_confidence`: mark wrong high-confidence rows for review or
  downweighting.
- `popularity_or_echo_overconfident`: identify head/echo-risk confidence that
  may amplify exposure bias.
- `grounding_uncertain`: separate title-to-catalog uncertainty from preference
  uncertainty.
- `ungrounded_high_confidence_noise_candidate`: a conservative prune-candidate
  scaffold for high-confidence ungrounded rows.

This is not a final pruning policy. It deliberately preserves hard tail
positives and avoids naive "high uncertainty means delete" behavior.

## Tests

Run the dedicated tests:

```powershell
python -m pytest tests/test_echo_simulation_triage.py
```

Run the full suite before committing substantial changes:

```powershell
python -m pytest
```

## Claim Guardrails

- Synthetic exposure simulation is a diagnostic scaffold.
- Triage reason codes and suggested weights are not proof of noise detection.
- Triage selective-risk diagnostics are decision-support metadata, not
  exposure-utility evidence.
- No output from this layer should be reported as a method result without
  approved observation artifacts, training/evaluation protocol, and complete
  manifests.
