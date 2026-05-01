# Data Format

## Interactions

Canonical interaction record:

```json
{
  "user_id": "u1",
  "item_id": "i9",
  "timestamp": 1234567890,
  "rating": 5.0,
  "domain": "movies"
}
```

Required fields: `user_id`, `item_id`. Timestamps are required for temporal
splits and strongly preferred for real experiments.

## Items

Canonical item record:

```json
{
  "item_id": "i9",
  "title": "Item Title",
  "description": "optional text",
  "category": "optional category",
  "brand": "optional brand",
  "domain": "movies",
  "raw_text": "optional source text"
}
```

Titles are required for generative recommendation and grounding.

## Processed examples

Canonical processed example:

```json
{
  "user_id": "u1",
  "history": ["i1", "i2"],
  "target": "i9",
  "candidates": ["i3", "i9", "i12"],
  "domain": "movies",
  "split": "test",
  "metadata": {"target_index": 3}
}
```

History must contain only interactions available before the target.

## Candidate sets

Candidate set records should include:

```json
{
  "user_id": "u1",
  "example_id": "u1:3",
  "candidate_items": ["i3", "i9", "i12"],
  "protocol": "full | sampled | loaded | retriever_generated",
  "seed": 13,
  "metadata": {}
}
```

## Predictions

Predictions follow the unified schema:

```json
{
  "user_id": "u1",
  "target_item": "i9",
  "candidate_items": ["i3", "i9", "i12"],
  "predicted_items": ["i9", "i12", "i3"],
  "scores": [0.9, 0.5, 0.1],
  "method": "ours_uncertainty_guided",
  "domain": "movies",
  "raw_output": null,
  "metadata": {}
}
```

## Metrics

Metrics are saved as:

- `metrics.json`: nested metric object;
- `metrics.csv`: flattened `scope,metric,value` rows;
- optional table/plot CSVs under `outputs/tables/` or run artifacts.

## Raw LLM outputs

Raw LLM output records should preserve:

- request ID or example ID;
- prompt template ID and hash;
- provider/model;
- request parameters;
- raw response text;
- parsed payload;
- token usage and latency;
- cache status;
- failure or retry metadata.

Raw outputs must not contain API keys or secrets.

## Adapter plan

### Tiny

Committed fixture adapter for tests and smoke runs. Not paper evidence.

### MovieLens-style

Map ratings to interactions, movie titles/genres to items, and timestamps to
temporal or leave-one-out examples. Use as local pilot before larger domains.

### Amazon Reviews-style multi-domain

Map reviews to interactions, product metadata to item records, and categories
to domains. Use train-only popularity and preserve raw file provenance. Do not
download data in Phase 7.

### Generic CSV/JSONL

Require field mapping for user/item/timestamp/rating/domain and item title/text
fields. The adapter should fail clearly if required fields are missing.
