# Phase 2A Observation Pipeline

Phase 2A builds the no-paid-API generative observation scaffold. It verifies
file formats, prompting, parsing, grounding, correctness labeling, metrics, and
resume behavior with `provider=mock`.

This is not an API pilot and not a paper result.

## File Flow

```text
processed examples
  -> observation inputs
  -> raw responses
  -> parsed predictions
  -> grounded predictions
  -> metrics/report/manifest
```

Default paths:

```text
data/processed/<dataset>/<processed_suffix>/
outputs/observation_inputs/<dataset>/<processed_suffix>/
outputs/observations/mock/<dataset>/<processed_suffix>/<run_name>/
```

`outputs/` is gitignored.

## Build Observation Inputs

```powershell
python scripts/build_observation_inputs.py --dataset movielens_1m --processed-suffix sanity_50_users --split test --max-examples 20 --stratify-by-popularity
```

Each JSONL record includes:

- user id;
- history item ids and titles;
- target item id and title;
- target timestamp;
- target popularity and bucket;
- prompt template name;
- prompt text;
- prompt hash;
- source processed/catalog paths.

The default template is `forced_json`, which asks for one generated item title,
yes/no self-verification, and confidence in `[0, 1]`.

## Run Mock Observation

```powershell
python scripts/run_observation_pipeline.py --provider mock --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --max-examples 20
```

The mock provider never calls external services and never requires an API key.
It supports:

- `oracle-ish`: sometimes returns the target title;
- `popularity_biased`: favors popular catalog titles;
- `random`: chooses deterministic catalog titles for sanity checks.

Outputs:

- `raw_responses.jsonl`
- `grounded_predictions.jsonl`
- `metrics.json`
- `report.md`
- `manifest.json`

The runner supports resume by skipping already completed `input_id` values.

## Metrics

The mock runner computes:

- GroundHit;
- correctness;
- ECE;
- Brier score;
- CBU_tau;
- WBC_tau;
- head/mid/tail confidence;
- head/mid/tail correctness;
- Tail Underconfidence Gap;
- wrong-high-confidence count;
- correct-low-confidence count.

These metrics are only sanity metrics when `provider=mock`. They must not be
reported as model behavior or paper evidence.

## Transition To Real API Pilot

Before a real API pilot, add provider adapters with:

- environment-variable API keys;
- rate limits and concurrency controls;
- retries and cache keys;
- idempotent resume;
- token/cost accounting where available;
- clear separation of prompt, raw response, parsed prediction, grounded
  prediction, and metrics.

Do not call a paid API until the provider is explicitly configured and the
pilot subset is approved.

## Phase 2B API Framework Flow

Phase 2B adds the real-provider framework, still dry-run by default:

```text
observation inputs
  -> API request records
  -> raw responses
  -> parsed predictions
  -> grounded predictions
  -> metrics/report/manifest
```

The API framework writes:

- `request_records.jsonl`
- `raw_responses.jsonl`
- `parsed_predictions.jsonl`
- `failed_cases.jsonl`
- `grounded_predictions.jsonl`
- `metrics.json`
- `report.md`
- `manifest.json`

Default dry-run command:

```powershell
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --max-examples 5 --dry-run
```

Dry-run responses are deterministic placeholders used to test cache, resume,
parsing, grounding, and output schemas. They are not API results. Real API
execution requires `--execute-api`, confirmed provider endpoint/model config,
and the corresponding API key environment variable.
