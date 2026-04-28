# Grounding Diagnostics

Grounding diagnostics audit catalog title ambiguity and optional observation
grounding margins before scaling API spend or treating pilot outputs as stable.
They are diagnostic artifacts only, not paper evidence.

## Why This Exists

Title-level generative recommendation depends on:

```text
generated title -> catalog item -> correctness/confidence analysis
```

If the catalog contains duplicate normalized titles, or if a generated title's
top grounding candidate is barely ahead of the second candidate, then
confidence and correctness analysis can be polluted by grounding uncertainty.
This diagnostic step makes those risks explicit.

## Catalog-Only Audit

Run on a processed dataset catalog:

```powershell
python scripts/analyze_grounding_diagnostics.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k
```

Outputs are ignored under:

```text
outputs/grounding_diagnostics/<dataset>/<processed_suffix>/
```

Files:

- `grounding_diagnostics_summary.json`
- `duplicate_title_groups.jsonl`
- `low_margin_cases.jsonl`
- `grounding_diagnostics_report.md`
- `grounding_diagnostics_manifest.json`

The catalog summary reports:

- empty title count;
- duplicate normalized title group count;
- number and fraction of items inside duplicate groups;
- bucket counts for catalog items;
- top duplicate title groups with item ids, titles, popularity, and buckets.

## Observation Margin Audit

If a pilot or mock run has `grounded_predictions.jsonl`, include it:

```powershell
python scripts/analyze_grounding_diagnostics.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --grounded-jsonl outputs/api_observations/deepseek/amazon_reviews_2023_beauty/sample_5k/test_forced_json_api_pilot30_parallel_20260428/grounded_predictions.jsonl --manifest-json outputs/api_observations/deepseek/amazon_reviews_2023_beauty/sample_5k/test_forced_json_api_pilot30_parallel_20260428/manifest.json
```

The observation margin audit reports:

- grounding status counts;
- rows with top-two candidate scores;
- low-margin rows under `--margin-threshold`;
- low-margin cases with generated title, target title, confidence,
  correctness, grounding status, and top candidates.

## Interpretation Guardrails

- Duplicate normalized titles indicate catalog ambiguity risk.
- Low candidate margin indicates grounding instability risk.
- These diagnostics do not prove model behavior or recommendation quality.
- Mock, dry-run, and pilot outputs must retain their source status.
- Before scaling API calls, inspect duplicate groups and low-margin cases to
  decide whether title normalization, catalog metadata, or grounding thresholds
  need revision.
