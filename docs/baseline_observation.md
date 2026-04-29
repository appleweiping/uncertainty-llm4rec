# Baseline Observation Interface

Baseline observation is the first reviewer-proofing layer after API and mock
observation scaffolds. It keeps the same title-level file contract:

```text
observation inputs
  -> baseline raw/parsed predictions
  -> generated title grounding
  -> grounded predictions
  -> metrics/report/manifest
```

This is not heavy recommender training and not a paper result. It exists so the
pipeline can compare DeepSeek/API observations with deterministic non-LLM
baselines under the same grounding and confidence-analysis schema.

## Implemented Baselines

- `popularity`: recommends the most popular catalog title not already in the
  user's history. The confidence proxy is normalized log popularity.
- `cooccurrence`: builds item-to-next-item counts from `split=train`
  observation examples only, aggregates counts from the user's history with a
  recency weight, and falls back to popularity when no train co-occurrence is
  available.

Both baselines generate an item title, not a ranked-list metric. The generated
title is grounded back to the catalog before correctness, GroundHit, ECE,
Brier, head/mid/tail summaries, and risk counts are computed.

## Command

Run on any existing observation input JSONL:

```powershell
python scripts/run_baseline_observation.py --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline popularity --max-examples 30
python scripts/run_baseline_observation.py --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline cooccurrence --max-examples 30
```

Outputs are ignored under:

```text
outputs/observations/baselines/<dataset>/<processed_suffix>/<input_stem>_<baseline>/
```

Files:

- `raw_responses.jsonl`
- `parsed_predictions.jsonl`
- `grounded_predictions.jsonl`
- `metrics.json`
- `report.md`
- `manifest.json`

## Guardrails

- No API is called.
- No API key is read.
- No model is trained.
- Co-occurrence counts are built from train-split observation examples only.
- Confidence is a simple proxy, not calibrated model confidence.
- Outputs are sanity/baseline artifacts until a full protocol run is explicitly
  executed and documented.

## Next Extensions

Later baseline phases can add ranking-to-title adapters for SASRec, BERT4Rec,
GRU4Rec, LightGCN, P5/TIGER/BIGRec-like systems, and Qwen3-8B server inference.
Each must write the same grounded prediction schema before entering analysis.
