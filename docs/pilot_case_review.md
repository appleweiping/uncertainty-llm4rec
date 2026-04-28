# Pilot Case Review

Pilot case review is a diagnostic layer for Storyflow observation outputs. It
does not create paper evidence by itself. Its job is to inspect small pilot
runs before scaling API spend or moving to full datasets.

## Purpose

The review answers practical questions after an observation run:

- Did the provider return parseable title-level JSON?
- Did generated titles ground to catalog items?
- Are wrong recommendations mostly high-confidence or low-confidence?
- Does the model say "yes" on wrong recommendations?
- Does the grounded generated item look more popular than the target item?
- Are errors concentrated in head/mid/tail buckets?
- Which concrete user histories need prompt or grounding inspection?

## Inputs

The script reads the standard API observation output directory:

```text
grounded_predictions.jsonl
failed_cases.jsonl
manifest.json
```

It also joins `manifest.input_jsonl` when available so the review can include
the user's history-title tail and the processed catalog path.

## Command

```powershell
python scripts/review_observation_cases.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_api_pilot20_non_thinking_20260428
```

Default outputs are ignored by git under:

```text
outputs/case_reviews/<source-run-relative-path>/
```

The files are:

- `case_review_summary.json`
- `case_review_cases.jsonl`
- `case_review.md`
- `case_review_manifest.json`

## Taxonomy

Primary failure types include:

- `parse_failure`
- `provider_failure`
- `ungrounded_high_confidence`
- `ungrounded_intermediate_confidence`
- `ungrounded_low_confidence`
- `wrong_high_confidence`
- `wrong_intermediate_confidence`
- `wrong_low_confidence`
- `correct_high_confidence`
- `correct_intermediate_confidence`
- `correct_low_confidence`

Overlay tags include:

- `self_verified_wrong`
- `grounding_failure`
- `fuzzy_grounding`
- `grounding_ambiguous`
- `generated_more_popular_than_target`
- `generated_head_target_tail`
- `wrong_high_confidence_generated_head`
- `correct_low_confidence_tail`
- target and generated popularity bucket tags.

These labels are for triage. A small pilot taxonomy must not be reported as a
general model behavior claim.

## How To Use The Review

Use the priority cases to decide the next engineering step:

- Many parse failures: adjust prompt or parser before more API calls.
- Many ungrounded high-confidence cases: inspect catalog/title normalization.
- Many wrong high-confidence generated-head cases: inspect popularity coupling.
- Many correct low-confidence tail cases: inspect tail underconfidence.
- Many fuzzy/ambiguous cases: improve grounding before scaling.

The review is allowed for mock, dry-run, smoke, and pilot outputs. Its report
must always preserve the source run's status instead of upgrading a pilot into
a full result.
