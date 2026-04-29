# Observation Analysis And Run Registry

Phase 2C adds analysis utilities for grounded generative observation outputs.
This layer reads completed mock, dry-run, pilot, or future full observation
runs and writes reproducible summaries without changing the raw run artifacts.

Mock and dry-run analysis is only a schema/analysis sanity check. It is not API
evidence, model behavior, or a paper result.

## Input Contract

The analysis command expects an observation run directory with:

```text
grounded_predictions.jsonl
failed_cases.jsonl        # optional for mock, expected for API runner
manifest.json             # optional but recommended
```

The grounded rows should include:

- generated title;
- confidence;
- correctness;
- target item title/id;
- target popularity and popularity bucket;
- grounded item id;
- grounding status and score;
- provider/model/dry-run metadata where available.

## Command

Analyze an API dry-run or future pilot run:

```powershell
python scripts/analyze_observation.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_dry_run
```

Analyze a mock observation run:

```powershell
python scripts/analyze_observation.py --run-dir outputs/observations/mock/movielens_1m/sanity_50_users/test_forced_json_popularity_biased
```

Use explicit files when a run directory uses a non-standard layout:

```powershell
python scripts/analyze_observation.py --grounded-jsonl path/to/grounded_predictions.jsonl --failed-jsonl path/to/failed_cases.jsonl --manifest-json path/to/manifest.json
```

For catalog-constrained or retrieval-context prompt gates, the command reads
the source `input_jsonl` from `manifest.json` when available. If a non-standard
run has no manifest, pass it explicitly:

```powershell
python scripts/analyze_observation.py --grounded-jsonl path/to/grounded_predictions.jsonl --input-jsonl path/to/observation_inputs.jsonl
```

## Outputs

By default, outputs are written under `outputs/analysis/...`, which is ignored
by git:

- `analysis_summary.json`
- `reliability_diagram.json`
- `bucket_summary.json`
- `repeat_summary.json`
- `candidate_diagnostic_summary.json`
- `candidate_diagnostic_cases.jsonl`
- `risk_cases.jsonl`
- `report.md`
- `analysis_manifest.json`

The command also appends a local pointer to:

```text
outputs/run_registry/observation_runs.jsonl
```

The registry is an ignored local index. It points to analysis artifacts and
source run paths; it is not a paper result table.

## Metrics And Slices

The analysis layer reports:

- GroundHit;
- correctness;
- mean confidence;
- ECE;
- Brier score;
- CBU_tau;
- WBC_tau;
- Tail Underconfidence Gap;
- reliability diagram bins overall and by head/mid/tail bucket;
- head/mid/tail confidence, correctness, and grounding summaries;
- repeat-target slices for `non_repeat_target`, `repeat_target`,
  `same_timestamp_repeat_target`, and missing repeat metadata;
- candidate-prompt diagnostics for retrieval-context/catalog-constrained gates;
- wrong-high-confidence cases;
- correct-low-confidence cases;
- grounding failure cases;
- parse failure summary;
- exploratory popularity-confidence slope.

The slope is a lightweight standard-library diagnostic over
`confidence ~ log1p(target_popularity)` plus a correctness-residualized variant.
It is a sanity diagnostic, not a causal claim. Causal popularity deconfounding
belongs to later CURE/TRUCE framework work.

Repeat-target slices are mandatory for Amazon Beauty full analysis because
repeat purchases and duplicate review artifacts can place the held-out target
item in the user's history. Use `non_repeat_target` as the cleaner ordinary
next-item probe and `repeat_target` as a separate diagnostic for repetition or
history-copy behavior. Do not collapse these slices into one paper claim without
reporting the sensitivity.

Candidate diagnostics join each grounded output back to the source observation
input and report:

- whether candidate context exists;
- whether the grounded generated item is in the provided candidate set;
- selected candidate rank, score, and head/mid/tail bucket;
- target-in-candidates and target-excluded counts;
- whether the grounded item copies an item from the user's history;
- confidence by selected candidate bucket.

These diagnostics are required for leakage-safe retrieval-context runs. If the
candidate policy excludes the held-out target, target-hit correctness is not a
recommendation-accuracy metric; the run measures prompt/candidate/grounding
behavior and confidence, not ordinary next-item success.

## Pilot And Full Run Use

For a future real API pilot, run analysis only after:

1. the API run has a manifest;
2. raw/parsed/grounded/failed files are present;
3. cache/resume status is recorded;
4. the run is clearly labeled as pilot, not full.

For full runs, every plot or table must trace back to:

- source observation manifest;
- analysis manifest;
- git commit hash;
- dataset config;
- provider/model config;
- command log.

## Case Review

Small smoke and pilot runs should also get a case-review pass before scale-up:

```powershell
python scripts/review_observation_cases.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_api_pilot20_non_thinking_20260428
```

Outputs are written under `outputs/case_reviews/...`:

- `case_review_summary.json`
- `case_review_cases.jsonl`
- `case_review.md`
- `case_review_manifest.json`

The review joins the source observation input, catalog, generated title,
grounded item, target title, confidence, target/generated popularity buckets,
and the user's history-title tail. It provides failure taxonomy labels such as
`wrong_high_confidence`, `ungrounded_high_confidence`,
`correct_low_confidence`, `self_verified_wrong`,
`generated_more_popular_than_target`, and `grounding_ambiguous`.

This taxonomy is a debugging tool for prompt/grounding/API scale-up decisions.
It must not be quoted as a paper result unless it comes from an approved full
run with complete manifests and the paper explicitly labels the scope.
