# Observation Analysis Plan

This plan explains how to analyze uncertainty-aware generative recommendation
after real runs finish. No fake examples or numbers should be added.

## Required artifacts

- `predictions.jsonl`
- `metrics.json`
- `reliability_diagram.csv`
- `risk_coverage.csv`
- `confidence_by_popularity_bucket.csv`
- case-study samples
- raw LLM outputs when available
- resolved configs and logs

## Questions

### 1. Does confidence correlate with correctness?

Use grounded correctness when available. Report ECE, Brier score, confidence
bucket accuracy, and confidence means for correct/incorrect examples.

### 2. Are wrong generations low-confidence or high-confidence?

Count and inspect:

- high confidence + correct;
- low confidence + correct;
- low confidence + wrong;
- high confidence + wrong.

High-confidence wrong and high-confidence hallucination cases should be sampled
for qualitative review.

### 3. Does confidence correlate with popularity?

Measure confidence by train-popularity bucket, correlation with log train
popularity, and overconfidence gaps for head/mid/tail items.

### 4. Are correct tail recommendations under-confident?

Filter to grounded correct examples and compare confidence for head/mid/tail
targets. Report under-confidence rate using a predeclared threshold.

### 5. Does high confidence correlate with low diversity/high history similarity?

Use history similarity, category repetition, novelty, and diversity metrics.
Compare high-confidence examples against lower-confidence examples.

### 6. Can uncertainty support abstention or fallback?

Use risk-coverage curves, fallback decision metadata, and accept/fallback
outcomes. Do not claim improvement until main metrics support it.

### 7. Can uncertainty prune noisy pseudo-labels later?

This is a later training question. Phase 7 should only specify needed artifacts:
generated title, grounded item, confidence, correctness, popularity bucket,
history similarity, and decision metadata.

## Outputs

Planned outputs:

- aggregate uncertainty table;
- reliability diagram data;
- risk-coverage data;
- confidence-by-popularity data;
- failure-case JSONL/CSV;
- qualitative case-study packet.

## Popularity, long-tail, and echo-risk analysis plan

Popularity analysis should use train-only popularity buckets and report
confidence, correctness, hallucination, and overconfidence by head/mid/tail
bucket. Long-tail analysis should focus on tail coverage, correct-tail
under-confidence, and abstention/fallback rates by bucket. Echo-risk analysis
should use history similarity, category repetition, diversity, and novelty
proxies, with high-confidence cases inspected separately. TBD: fill from real
metrics files.

## Evidence after R3 / R3b

R3 offline refinement supports **observation-first** claims: miscalibration,
high-confidence wrong behavior, grounding and candidate adherence failures,
parse failures, and accept/fallback/rerank decision attribution. It does **not**
support a primary claim that **OursMethod improves ranking** over fallback-only.

R3b conservative gate cache replay (`r3b_movielens_1m_conservative_gate_cache_replay`)
validates the conservative configuration against the **same** R3 cache without
live API calls; export `outputs/tables/r3b_conservative_gate_*.csv` and
`r3b_observation_failures.csv` via `scripts/export_r3b_tables.py` after runs
complete. Do not scale to multi-dataset **method** comparisons in this gate;
multi-dataset **observation** studies are a separate, explicitly scoped track.
