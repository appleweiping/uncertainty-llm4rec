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
  -> analysis summary/reliability/risk cases/run registry
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

For approved real API runs, request records and the manifest also include
`run_label`, `budget_label`, and `execution_mode`. Cache keys include
`execution_mode`, so a dry-run response cannot be reused as a real API response.
The runner can process multiple in-flight requests with `--max-concurrency`,
while a global `--rate-limit` controls request start rate.

Phase 2C reads these outputs without mutating them:

- `analysis_summary.json`
- `reliability_diagram.json`
- `bucket_summary.json`
- `risk_cases.jsonl`
- `report.md`
- `analysis_manifest.json`
- `outputs/run_registry/observation_runs.jsonl`

Pilot case review is a separate diagnostic layer:

- `case_review_summary.json`
- `case_review_cases.jsonl`
- `case_review.md`
- `case_review_manifest.json`

Default dry-run command:

```powershell
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --max-examples 5 --dry-run
```

Dry-run responses are deterministic placeholders used to test cache, resume,
parsing, grounding, and output schemas. They are not API results. Real API
execution requires `--execute-api`, confirmed provider endpoint/model config,
the corresponding API key environment variable, and an approved provider/model
/budget/rate/concurrency gate.

Approved accelerated pilot command shape:

```powershell
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --sample-size 30 --stage pilot --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 30 --approved-max-concurrency 3 --approved-budget-label USER_APPROVED_BEAUTY_SAMPLE30_PARALLEL --execute-api-intended --allow-over-20
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --max-examples 30 --execute-api --rate-limit 30 --max-concurrency 3 --run-label amazon_beauty_sample30_deepseek_parallel --budget-label USER_APPROVED_BEAUTY_SAMPLE30_PARALLEL
```

This remains a pilot flow. Full API observation requires a separate manifest,
budget label, and scale-up decision.

## Analyze Observation Outputs

Run analysis on a completed mock/dry-run/pilot output directory:

```powershell
python scripts/analyze_observation.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_dry_run
```

The analysis layer reports reliability bins, head/mid/tail confidence and
correctness, wrong-high-confidence cases, correct-low-confidence cases,
grounding failures, parse failures, and an exploratory popularity-confidence
slope. Analysis of mock or dry-run outputs is still only a schema sanity
artifact and not paper evidence.

Run case review on an approved smoke/pilot output directory:

```powershell
python scripts/review_observation_cases.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_api_pilot20_non_thinking_20260428
```

Case review joins the observation input history, generated title, grounded
catalog item, target title, confidence, and target/generated popularity
buckets. It is meant for prompt/grounding triage before scale-up, not for
paper claims.
