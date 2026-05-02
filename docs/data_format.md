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
splits and strongly preferred for real experiments. Raw interaction file paths,
processed interaction paths, and field mappings must be declared in the dataset
config or real-experiment template before any pilot run.

User and item IDs may be remapped internally, but the mapping must be saved with
processed artifacts and reused consistently for items, interactions, candidates,
predictions, and metrics. Timestamp handling must be explicit: temporal splits
use the configured timestamp field; leave-one-out splits must document the order
used when timestamps are missing.

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

Processed examples must record the split strategy and candidate protocol used to
create them. Repeat-target or no-repeat-target slicing must be declared in the
config; if both are analyzed, they must be reported as separate protocols.

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

Main ranking protocols must include the held-out target item in the candidate
set. Candidate protocols that deliberately exclude the target are allowed only
for grounding, retrieval-context, prompt, or other diagnostics and must not be
interpreted as recommendation accuracy. Candidate sets used by comparable
baselines and OursMethod must be identical or explicitly documented as a
separate comparison. Saved candidate-set paths are required for real runs unless
the full-ranking candidate set is reconstructible from the resolved config and
catalog snapshot.

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

## Real experiment template fields

Before a real pilot, each template must declare:

- raw data path and processed data path, or `TBD` while still a template;
- interaction and item metadata schema;
- user/item ID mapping policy and saved mapping location;
- timestamp handling and split strategy;
- temporal or leave-one-out policy;
- negative sampling policy, including seed and sampling pool;
- full-ranking vs sampled-ranking protocol;
- candidate set construction and saved candidate-set path;
- train-only popularity source for buckets, novelty, and long-tail metrics;
- domain field for multi-domain runs;
- repeat-target or no-repeat-target slicing policy;
- whether any target-excluding candidate protocol is diagnostic only.

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
