# Calibration Analysis Plan

Verbalized confidence alone is not claimed as novel. It is one uncertainty
signal among grounding, candidate-normalized confidence, popularity, and
history-similarity signals.

## Metrics

- ECE.
- Brier score.
- Reliability diagram.
- Confidence bucket accuracy.
- Risk-coverage curve.
- High-confidence wrong rate.
- Low-confidence correct rate.
- AUROC/AUPRC for confidence predicting correctness if implemented later.

## Correctness target

Use `metadata.is_grounded_hit` when present. Otherwise use top-1 exact item
match from the unified prediction schema. Report the chosen target.

## Reliability diagram

Export bucket-level rows with:

- bin index;
- lower/upper confidence;
- count;
- mean confidence;
- empirical accuracy.

## Risk-coverage

Sort examples by confidence descending and compute coverage and risk at each
prefix. Use this to evaluate abstention or selective prediction policies.

## High-confidence wrong rate

Define a threshold before analysis, for example `confidence >= 0.85`. Count
wrong grounded predictions and hallucinations in this subset.

## Low-confidence correct rate

Define a threshold before analysis, for example `confidence < 0.7`. Count
grounded correct predictions in this subset, especially tail items.

## Later AUROC/AUPRC

If implemented, treat confidence as a score for predicting correctness. Report
AUROC/AUPRC only from real completed predictions with enough sample size.

## Artifact inputs

- `predictions.jsonl`
- `metrics.json`
- `reliability_diagram.csv`
- `risk_coverage.csv`
- `confidence_by_popularity_bucket.csv`

## Evidence after R3 / R3b

R3 MovieLens candidate-500 artifacts show **strong miscalibration** (high ECE /
Brier relative to a well-calibrated model) and frequent **high-confidence wrong**
generations under verbalized confidence. These support calibration-focused
**observation** sections; they do **not** justify claims that raw LLM confidence
is decision-ready without grounding, adherence checks, or conservative routing.

R3b tables (`r3b_conservative_gate_main.csv`, `r3b_observation_failures.csv`)
aggregate the same diagnostics across seeds under **cache-only** replay; fill
paper text from those files only after the gate run finishes.

## Rerank parser follow-up (no live API yet)

R3 found `llm_rerank_real` near **zero-hit** behavior driven largely by
**truncated JSON / parse failure**, not evaluator bugs. A narrow parser recovery
for closed `ranked_items` arrays is already in tree; **do not** rerun live rerank
API until a small **rerank-only** subgate is approved. Prefer **cache replay**
from saved raw outputs; if raw outputs cannot reconstruct scores, plan a
minimal new API batch under `docs/server_runbook.md` safeguards.
