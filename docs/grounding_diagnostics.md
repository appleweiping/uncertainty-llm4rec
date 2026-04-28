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
- `grounding_failure_cases.jsonl`
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

The failure taxonomy additionally reviews non-grounded predictions. If the run
does not already contain grounding candidates, the diagnostic recomputes a
lightweight catalog top-candidate similarity for review only. It does not
change the official grounded prediction file.

Failure cases are tagged with:

- `near_miss_candidate`: top catalog candidate score is above
  `--near-miss-threshold`;
- `weak_candidate_overlap`: the generated title has some catalog overlap but
  not enough for a near miss;
- `no_catalog_support`: no useful catalog candidate is found;
- `generated_title_too_generic`: the generated title is too short or generic;
- `duplicate_title_risk`: normalized generated title collides with duplicated
  catalog titles;
- `target_title_near_generated`: target title and generated title are similar
  but grounding still failed;
- `high_confidence_ungrounded`: confidence is above
  `--high-confidence-threshold` while grounding failed.

The output includes `recommended_actions` for prompt review, catalog-constrained
gates, threshold/normalization inspection, and duplicate-title disambiguation.
These are triage labels, not paper conclusions.

## Interpretation Guardrails

- Duplicate normalized titles indicate catalog ambiguity risk.
- Low candidate margin indicates grounding instability risk.
- Failure taxonomy labels indicate engineering follow-up categories.
- These diagnostics do not prove model behavior or recommendation quality.
- Mock, dry-run, and pilot outputs must retain their source status.
- Before scaling API calls, inspect duplicate groups and low-margin cases to
  decide whether title normalization, catalog metadata, grounding thresholds,
  or prompt context need revision.
