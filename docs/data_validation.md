# Processed Data Validation

This document describes the Phase 1 processed-dataset audit. Validation checks
local processed outputs before any generative observation run. It does not
produce experiment results.

## Command

```powershell
python scripts/validate_processed_dataset.py --dataset movielens_1m --processed-suffix sanity_50_users
```

Default outputs are written under:

```text
outputs/data_validation/<dataset>/<processed_suffix>/
```

The script writes:

- `validation_summary.json`
- `validation_report.md`

Both paths are ignored by git through `outputs/`.

## Checks

The validator checks that the processed directory contains:

- `item_catalog.csv`
- `interactions.csv`
- `item_popularity.csv`
- `user_sequences.jsonl`
- `observation_examples.jsonl`
- `preprocess_manifest.json`

It also checks that observation examples contain the fields needed by
title-level generative recommendation:

- user id;
- history item ids;
- history item titles;
- target item id;
- target item title;
- split;
- target timestamp;
- target item popularity, also aliased as item popularity;
- target popularity bucket, also aliased as popularity bucket.

Chronological checks verify that each example history is a prefix before the
target index and that history timestamps do not exceed the target timestamp. In
the per-user `leave_last_two` setting, the validator checks that each user has
one validation target at the second-to-last item and one test target at the
last item.

The validator also checks title quality, duplicate normalized catalog titles,
non-empty head/mid/tail target buckets, and manifest provenance fields such as
data scale, filtering parameters, split policy, generation time, and config
snapshot.

## Exit Behavior

- Blockers produce a non-zero exit code.
- Warnings keep exit code zero but are written clearly to JSON and markdown.
- Passing validation means the processed data is structurally ready for
  observation input construction. It does not mean an API pilot or experiment
  has run.
