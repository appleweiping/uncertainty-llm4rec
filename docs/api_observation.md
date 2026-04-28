# API Observation Framework

Phase 2B adds the API observation framework in dry-run mode. It prepares the
provider, cache, resume, parsing, and output structure needed for a later
approved API pilot. It does not call real APIs by default. The current active
large-model API target is DeepSeek only; multi-provider API runs are deferred.

## Secrets

Copy `.env.example` to a local `.env` only when a real pilot is approved:

```text
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=
MOONSHOT_API_KEY=
ZHIPUAI_API_KEY=
```

Never commit `.env`, API keys, raw sensitive API responses, or provider cache
containing private content.

## Provider Configs

Provider configs live under `configs/providers/`:

- `deepseek.yaml`
- `qwen_api.yaml`
- `kimi.yaml`
- `glm.yaml`

DeepSeek is the current single-provider pilot target:

- provider: `deepseek`;
- model: `deepseek-v4-flash`;
- base URL: `https://api.deepseek.com`;
- endpoint: `/chat/completions`;
- API key env: `DEEPSEEK_API_KEY`.

Future provider configs remain placeholders until the user confirms exact
provider/model settings. Each config defines:

- provider name;
- model name placeholder;
- API key environment variable;
- base URL / endpoint placeholder;
- timeout;
- max concurrency;
- retry/backoff;
- rate limit;
- cache policy;
- dry-run default.

The real API runner will refuse execution while placeholder fields remain or
when `requires_endpoint_confirmation` is true.

## Dry-Run

Build observation inputs if needed:

```powershell
python scripts/build_observation_inputs.py --dataset movielens_1m --processed-suffix sanity_50_users --split test --max-examples 20 --stratify-by-popularity
```

Run API framework dry-run:

```powershell
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --max-examples 5 --dry-run
```

Dry-run never reads API keys and never makes a network call. It creates
deterministic fake provider responses so the request schema, cache, parser,
grounding, correctness labeling, metrics, and resume behavior can be checked.

Check DeepSeek smoke-test readiness without making a network call:

```powershell
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --sample-size 5 --stage smoke --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 10 --approved-budget-label USER_APPROVED_SMOKE --execute-api-intended
```

The readiness check prints whether required gates pass. It never prints the API
key value and never calls DeepSeek.

## Real API Pilot

A real API pilot must be explicitly approved with provider, model, budget, and
rate limits. The command shape is:

```powershell
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --max-examples 20 --execute-api --rate-limit 10 --max-concurrency 1
```

This will still fail unless:

- the provider config has confirmed endpoint/model values;
- the relevant API key environment variable exists;
- the user has approved the pilot.

For the first smoke test, the intended limits are:

- provider: DeepSeek only;
- sample size: 5;
- max concurrency: 1;
- rate limit: 10 requests/minute or lower;
- cache/resume enabled;
- run analysis immediately after completion.

## Cache And Resume

Cache keys are deterministic over:

```text
provider + model + prompt_template + temperature + prompt/input hash
```

Cache files are written under the provider config's cache directory, currently
`outputs/api_cache/...`, which is ignored by git. Resume skips input ids already
present in grounded predictions or failed cases.

To avoid duplicate paid calls in a future real pilot:

- keep cache enabled;
- use `--resume`;
- do not delete `outputs/api_cache/...` until the run is complete;
- run a small pilot before a full batch.

## Output Layers

The API runner writes:

- `request_records.jsonl`
- `raw_responses.jsonl`
- `parsed_predictions.jsonl`
- `failed_cases.jsonl`
- `grounded_predictions.jsonl`
- `metrics.json`
- `report.md`
- `manifest.json`

Raw responses are kept separate from parsed and grounded records. Parse
failures are written to `failed_cases.jsonl` and do not silently disappear.

## Analysis Layer

After a dry-run or approved pilot, generate analysis artifacts:

```powershell
python scripts/analyze_observation.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_dry_run
```

This writes ignored files under `outputs/analysis/...` and appends a local
pointer to `outputs/run_registry/observation_runs.jsonl`. The report includes
reliability bins, head/mid/tail summaries, wrong-high-confidence cases,
correct-low-confidence cases, grounding failures, parse failures, and a
lightweight popularity-confidence slope diagnostic. Dry-run analysis is not a
real API pilot and not paper evidence.

## Parsing

The parser attempts:

1. strict JSON;
2. fenced JSON code block;
3. embedded JSON object;
4. regex fallback.

Confidence can be `0.72`, `72`, `"72%"`, or similar formats that normalize to
`[0, 1]`. Yes/no variants are normalized. Empty generated titles are parse
failures.

## Pilot Versus Full

- Dry-run: framework sanity only, no API call.
- API pilot: approved small real API call, usually 20-50 MovieLens examples.
- Full run: larger provider/dataset run with explicit manifests, logs, cache,
  and budget controls.

No API pilot or full run has been executed by Codex in this repository.
