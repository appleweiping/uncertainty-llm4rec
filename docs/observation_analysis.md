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

## Outputs

By default, outputs are written under `outputs/analysis/...`, which is ignored
by git:

- `analysis_summary.json`
- `reliability_diagram.json`
- `bucket_summary.json`
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
- wrong-high-confidence cases;
- correct-low-confidence cases;
- grounding failure cases;
- parse failure summary;
- exploratory popularity-confidence slope.

The slope is a lightweight standard-library diagnostic over
`confidence ~ log1p(target_popularity)` plus a correctness-residualized variant.
It is a sanity diagnostic, not a causal claim. Causal popularity deconfounding
belongs to later CURE/TRUCE framework work.

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
