# API Observation Framework

Phase 2B adds the API observation framework in dry-run mode. It prepares the
provider, cache, resume, parsing, and output structure needed for a later
approved API pilot. It does not call real APIs by default.

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

Each config defines:

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

Endpoint and model names are deliberately marked with `TODO_CONFIRM...` until
the user confirms the exact provider/model settings. The real API runner will
refuse execution while these placeholders remain.

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
