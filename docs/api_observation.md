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

The API adapter uses `certifi` for TLS certificate verification. Do not disable
certificate verification to work around local CA issues.

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
- max tokens: `1024`, raised after the first diagnostic showed
  `deepseek-v4-flash` spent 256 completion tokens on `reasoning_content` and
  returned empty final content with `finish_reason=length`.
- extra body: `thinking.type=disabled`, because this observation task needs a
  short final JSON answer rather than reasoning-only output.

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
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --sample-size 5 --stage smoke --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 10 --approved-max-concurrency 1 --approved-budget-label USER_APPROVED_SMOKE --execute-api-intended
```

The readiness check prints whether required gates pass. It never prints the API
key value and never calls DeepSeek.

## Real API Pilot

A real API pilot must be explicitly approved with provider, model, budget,
rate limits, and concurrency. The conservative command shape is:

```powershell
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/movielens_1m/sanity_50_users/test_forced_json.jsonl --max-examples 20 --execute-api --rate-limit 10 --max-concurrency 1 --run-label movielens_deepseek_pilot --budget-label USER_APPROVED_PILOT
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

After smoke stability is verified, safe acceleration is allowed only through
explicit gate records. Use `--rate-limit` to cap global request starts per
minute and `--max-concurrency` to allow multiple in-flight requests. The runner
uses a thread-safe start limiter, incremental JSONL writes, cache/resume, and
per-request `run_label`, `budget_label`, and `execution_mode` metadata. A
sample accelerated gate looks like:

```powershell
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --sample-size 30 --stage pilot --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 30 --approved-max-concurrency 3 --approved-budget-label USER_APPROVED_BEAUTY_SAMPLE30_PARALLEL --execute-api-intended --allow-over-20
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --output-dir outputs/api_observations/deepseek/amazon_reviews_2023_beauty/sample_5k/test_forced_json_api_pilot30_parallel --max-examples 30 --execute-api --rate-limit 30 --max-concurrency 3 --run-label amazon_beauty_sample30_deepseek_parallel --budget-label USER_APPROVED_BEAUTY_SAMPLE30_PARALLEL
```

The readiness manifest must be kept with the run artifacts under ignored
`outputs/`. Do not scale beyond the approved sample/rate/concurrency without a
new budget label and manifest.

## Cache And Resume

Cache keys are deterministic over:

```text
provider + model + prompt_template + temperature + prompt/input hash + max_tokens
+ request_options + execution_mode
```

Cache files are written under the provider config's cache directory, currently
`outputs/api_cache/...`, which is ignored by git. Resume skips input ids already
present in grounded predictions or failed cases.

`execution_mode` is part of the cache key and cached responses are rejected when
their stored `dry_run` flag does not match the current run. This prevents a
dry-run response from being counted as a real API observation.

To avoid duplicate paid calls in a future real pilot:

- keep cache enabled;
- use `--resume`;
- do not delete `outputs/api_cache/...` until the run is complete;
- run a small pilot before a full batch.

For one-off provider diagnostics, `scripts/run_api_observation.py` supports
`--no-cache`. Use it only for an approved tiny diagnostic request; normal smoke
and pilot runs should keep cache enabled to avoid duplicate paid calls.

## Output Layers

The API runner writes:

- `request_records.jsonl`
- `raw_responses.jsonl` with extracted `raw_text` and ignored full provider
  `raw_payload` for debugging
- `parsed_predictions.jsonl`
- `failed_cases.jsonl`
- `grounded_predictions.jsonl`
- `metrics.json`
- `report.md`
- `manifest.json`

Raw responses are kept separate from parsed and grounded records and remain in
gitignored `outputs/`. Parse failures are written to `failed_cases.jsonl` and
do not silently disappear.

Grounded predictions also preserve repeat-target fields from the observation
input (`target_in_history`, `target_history_occurrence_count`,
`target_same_timestamp_as_history`, and history duplicate counts). For Amazon
Beauty full runs, use repeat-aware input files when reporting sensitivity to
repeat purchase or duplicate review artifacts:

```powershell
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --repeat-target-policy exclude
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --repeat-target-policy only
```

## Analysis Layer

After a dry-run or approved pilot, generate analysis artifacts:

```powershell
python scripts/analyze_observation.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_dry_run
```

This writes ignored files under `outputs/analysis/...` and appends a local
pointer to `outputs/run_registry/observation_runs.jsonl`. The report includes
reliability bins, head/mid/tail summaries, wrong-high-confidence cases,
correct-low-confidence cases, grounding failures, parse failures, and a
lightweight popularity-confidence slope diagnostic. It also writes
`repeat_summary.json` when grounded rows preserve repeat-target metadata, so
Amazon Beauty full reports can separate no-repeat ordinary next-item probes
from repeat-only diagnostics. Dry-run analysis is not a real API pilot and not
paper evidence.

For small real pilots, also generate case-review artifacts:

```powershell
python scripts/review_observation_cases.py --run-dir outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_api_pilot20_non_thinking_20260428
```

This produces an ignored failure taxonomy under `outputs/case_reviews/...`.
It records primary types such as `wrong_high_confidence`,
`ungrounded_high_confidence`, `correct_low_confidence`, parse/provider
failures, and overlay tags such as `self_verified_wrong`,
`generated_more_popular_than_target`, and `grounding_ambiguous`. The taxonomy
is for debugging and scale-up decisions; a pilot review is not a full result.

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
- API pilot: approved small real API call, usually 20-50 MovieLens or Amazon
  sample examples.
- Full run: larger provider/dataset run with explicit manifests, logs, cache,
  and budget controls.

The readiness checker supports `--stage smoke`, `--stage pilot`, and
`--stage full`. `full` has no default 20-example cap, but it still requires an
explicit provider/model/rate/concurrency/budget gate, `--execute-api-intended`,
an API key environment variable, cache enabled, and ignored outputs. Use it for
approved full-slice runs such as Amazon Beauty no-repeat test observation:

```powershell
python scripts/check_api_pilot_readiness.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl --sample-size 185 --stage full --approved-provider deepseek --approved-model deepseek-v4-flash --approved-rate-limit 30 --approved-max-concurrency 3 --approved-budget-label USER_APPROVED_BEAUTY_FULL_NOREPEAT_20260429 --execute-api-intended
python scripts/run_api_observation.py --provider-config configs/providers/deepseek.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl --output-dir outputs/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_api_full185_20260429 --max-examples 185 --execute-api --rate-limit 30 --max-concurrency 3 --run-stage full --run-label amazon_beauty_full_no_repeat_deepseek_20260429 --budget-label USER_APPROVED_BEAUTY_FULL_NOREPEAT_20260429
```

## Current Smoke Status

On 2026-04-28, the user approved DeepSeek smoke and pilot work with:

- provider: DeepSeek;
- model: `deepseek-v4-flash`;
- smoke sample size: 5;
- pilot sample size: 20;
- max concurrency: 1;
- rate limit: 10 requests/minute;
- budget label: `USER_APPROVED_SMOKE_20260428`.

The first real attempt failed at local TLS certificate verification. After
adding `certifi`, the retry reached the provider, but all five records returned
empty final content because the model spent the completion budget on
`reasoning_content`. Official DeepSeek docs state that the `thinking` parameter
can switch between thinking and non-thinking mode. The active config therefore
sets `thinking.type=disabled` for this short forced-JSON observation task.

After that fix, the 5-example smoke and 20-example pilot both produced parsed
and grounded predictions. The 20-example pilot is a small sanity pilot only,
not a full run or paper result. Ignored local outputs are under:

```text
outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_api_smoke_non_thinking_20260428/
outputs/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_api_pilot20_non_thinking_20260428/
outputs/analysis/api_observations/deepseek/movielens_1m/sanity_50_users/test_forced_json_api_pilot20_non_thinking_20260428/
```

On the same date, after Amazon Beauty sample readiness, the user approved a
30-example Amazon Beauty sample pilot with:

- provider: DeepSeek;
- model: `deepseek-v4-flash`;
- sample size: 30;
- max concurrency: 3;
- rate limit: 30 requests/minute;
- budget label: `USER_APPROVED_BEAUTY_SAMPLE30_PARALLEL_20260428`.

The first 30-example attempt exposed a dry-run cache contamination risk in the
runner and was treated as invalid diagnostic output. The runner was then fixed
to include `execution_mode` in cache keys and reject mismatched cached
responses. The corrected parallel pilot produced 30 parsed and grounded
records with no parse/provider failures, followed by analysis and case-review
diagnostics under ignored output paths:

```text
outputs/api_observations/deepseek/amazon_reviews_2023_beauty/sample_5k/test_forced_json_api_pilot30_parallel_20260428/
outputs/analysis/api_observations/deepseek/amazon_reviews_2023_beauty/sample_5k/test_forced_json_api_pilot30_parallel_20260428/
outputs/case_reviews/api_observations/deepseek/amazon_reviews_2023_beauty/sample_5k/test_forced_json_api_pilot30_parallel_20260428/
```

This is a local sample pilot only.

## Current Amazon Beauty Full-Slice Status

On 2026-04-29, the user-approved Amazon Beauty no-repeat full-slice observation
was executed with:

- provider: DeepSeek;
- model: `deepseek-v4-flash`;
- input: `outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl`;
- sample/input count: 185 repeat-free test examples;
- rate limit: 30 requests/minute;
- max concurrency: 3;
- run stage: `full`;
- budget label: `USER_APPROVED_BEAUTY_FULL_NOREPEAT_20260429`.

Artifacts are under ignored paths:

```text
outputs/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_api_full185_20260429/
outputs/analysis/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_api_full185_20260429/
outputs/case_reviews/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_api_full185_20260429/
```

The run completed with parsed/grounded outputs and no failed cases. It is a
scoped full-slice observation artifact, not a paper conclusion. Because the
free-form title-generation path shows substantial grounding and target-hit
risk on this catalog, the next gate should compare retrieval-context or
catalog-constrained prompts before any broader free-form API scale-up.

Build the leakage-safe no-API comparison inputs for that next gate with:

```powershell
python scripts/build_observation_gate_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix full --split test --max-examples 185 --stratify-by-popularity --candidate-count 20 --repeat-target-policy exclude
```

The command writes ignored `test_gate185_no_repeat_*` JSONL files and a gate
manifest. It does not call an API. The retrieval-context and
catalog-constrained variants exclude the held-out target by default, so they
are prompt/grounding diagnostics rather than recommendation-accuracy evidence.

## Current Amazon Beauty Retrieval-Context Diagnostic Status

On 2026-04-29, a user-approved DeepSeek retrieval-context diagnostic was run on
the same 185 repeat-free Amazon Beauty test inputs:

- provider: DeepSeek;
- model: `deepseek-v4-flash`;
- input:
  `outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_retrieval_context_json_c20.jsonl`;
- sample/input count: 185;
- rate limit: 30 requests/minute;
- max concurrency: 3;
- run stage: `full`;
- budget label:
  `USER_APPROVED_BEAUTY_FULL_NOREPEAT_RETRIEVAL_CONTEXT_RETRY_20260429`.

The first attempt was blocked by local network permission and wrote 185
provider-stage failures without raw responses. A retry with the same ignored
artifact policy completed with `failed_count=0` and `total_grounded_count=185`.
Artifacts are under ignored paths:

```text
outputs/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_retrieval_context_json_c20_api_full185_retry_20260429/
outputs/analysis/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_retrieval_context_json_c20_api_full185_retry_20260429/
outputs/case_reviews/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_retrieval_context_json_c20_api_full185_retry_20260429/
```

Key diagnostic values:

- count: 185;
- GroundHit: `0.973`;
- failed cases: 0;
- target correctness: `0.0`, but this is not a recommendation-accuracy metric
  because the target item is excluded from candidates by design;
- wrong-high-confidence count: 179;
- WBC_tau: `0.968`;
- candidate context rows: 185;
- generated-in-candidate-set count/rate: `162/185` = `0.876`;
- grounded-not-in-candidate-set count: 18;
- ungrounded-with-candidate-context count: 5;
- target-in-candidates count: 0;
- mean selected candidate rank: `5.586`;
- selected candidate buckets: head=49, mid=73, tail=40.

Compared with the free-form no-repeat slice, retrieval context substantially
improves catalog grounding while preserving a strong overconfidence signal.
The new candidate diagnostics show that most grounded titles are selected from
the provided candidate context, while target leakage is absent by construction.
This is a prompt/candidate/grounding diagnostic artifact, not a paper
conclusion or recommendation-accuracy result.

## Current Amazon Beauty Catalog-Constrained Diagnostic Status

On 2026-04-29, a matching user-approved DeepSeek catalog-constrained diagnostic
was run on the same 185 repeat-free Amazon Beauty test inputs:

- provider: DeepSeek;
- model: `deepseek-v4-flash`;
- input:
  `outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_catalog_constrained_json_c20.jsonl`;
- sample/input count: 185;
- rate limit: 30 requests/minute;
- max concurrency: 3;
- run stage: `full`;
- budget label:
  `USER_APPROVED_BEAUTY_FULL_NOREPEAT_CATALOG_CONSTRAINED_20260429`.

The gate order was readiness check, 5-example dry-run, 5-example real smoke,
then cache/resume full185. Artifacts are under ignored paths:

```text
outputs/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_catalog_constrained_json_c20_api_full185_20260429/
outputs/analysis/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_catalog_constrained_json_c20_api_full185_20260429/
outputs/case_reviews/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_gate185_no_repeat_catalog_constrained_json_c20_api_full185_20260429/
```

Key diagnostic values:

- count: 185;
- failed cases: 0;
- GroundHit: `0.784`;
- target correctness: `0.0`, but this is not recommendation accuracy because
  the target item is excluded from candidates by design;
- mean confidence: `0.658`;
- WBC_tau: `0.762`;
- wrong-high-confidence count: 141;
- ungrounded low-confidence count: 40;
- candidate context rows: 185;
- generated-in-candidate-set count/rate: `140/185` = `0.757`;
- target-in-candidates count: 0;
- mean selected candidate rank: `12.564`;
- selected candidate buckets: head=29, mid=40, tail=71.

Compared with retrieval-context, this catalog-constrained prompt is stricter
about choosing exactly from round-robin head/mid/tail candidates, but it leaves
more ungrounded `NO_GROUNDABLE_TITLE` cases and still preserves many
wrong-high-confidence selections among target-excluded candidates. It is a
prompt/candidate QA artifact, not a paper conclusion or method result.

## Current Local Amazon All-Domain Free-Form Status

On 2026-04-30, after the user approved DeepSeek provider/model and an
unlimited fast-but-controlled budget, the currently local processed Amazon
domains were run through the free-form forced-JSON observation path. This is
the local processed-domain scope only:

- included: Beauty, Health_and_Personal_Care, Handmade_Products,
  Digital_Music;
- not included: Amazon categories whose raw files are not yet local or not yet
  processed, such as Video_Games, Sports_and_Outdoors, and Books.

Execution settings:

- provider: DeepSeek;
- model: `deepseek-v4-flash`;
- rate limit: 60 requests/minute;
- max concurrency: 5;
- cache/resume: enabled;
- run stage: `full`;
- budget label:
  `USER_APPROVED_AMAZON_LOCAL_ALLDOMAIN_UNLIMITED_FAST_20260430`.

Before the real API calls, each domain passed the readiness checker and a
5-example dry-run smoke. The real runs completed with no failed API cases:

| Domain | Count | Failed | GroundHit | Target correctness | ECE | WBC_tau | Wrong-high confidence | Ungrounded-high confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beauty | 225 | 0 | 0.316 | 0.160 | 0.648 | 0.974 | 184 | 153 |
| Health_and_Personal_Care | 72 | 0 | 0.181 | 0.083 | 0.691 | 0.955 | 63 | 56 |
| Handmade_Products | 28 | 0 | 1.000 | 0.786 | 0.171 | 1.000 | 6 | 0 |
| Digital_Music | 10 | 0 | 1.000 | 1.000 | 0.080 | 0.000 | 0 | 0 |

Ignored artifact roots:

```text
outputs/api_observations/deepseek/amazon_reviews_2023_beauty/full/test_forced_json_full225_alldomain_20260430/
outputs/api_observations/deepseek/amazon_reviews_2023_health/full/test_forced_json_full72_alldomain_20260430/
outputs/api_observations/deepseek/amazon_reviews_2023_handmade/full/test_forced_json_full28_alldomain_20260430/
outputs/api_observations/deepseek/amazon_reviews_2023_digital_music/full/test_forced_json_full10_alldomain_20260430/
outputs/analysis/api_observations/deepseek/*/full/test_forced_json_full*_alldomain_20260430/
outputs/case_reviews/api_observations/deepseek/*/full/test_forced_json_full*_alldomain_20260430/
outputs/grounding_diagnostics/amazon_reviews_2023_*/full/
```

Interpretation guardrails:

- This is a real API observation artifact with manifests, not a trained method
  result or paper conclusion.
- Beauty and Health show strong free-form grounding and overconfidence risks,
  which supports the next gate: retrieval-context and catalog-constrained
  comparisons on Health before broader claims.
- Handmade_Products and Digital_Music are repeat-heavy under the current local
  5-core split. Their high target correctness mainly diagnoses repeat/history
  copying behavior and must not be generalized as recommendation accuracy.
- Confidence values are model-reported confidence, not calibrated
  exposure-counterfactual confidence.

## Current Health and Video_Games Prompt-Gate Status

On 2026-05-01, after the user explicitly approved DeepSeek API execution and
local runnable full-data gates, Health_and_Personal_Care and Video_Games were
run through matched prompt-shape gates. The execution settings were:

- provider: DeepSeek;
- model: `deepseek-v4-flash`;
- rate limit: 60 requests/minute;
- max concurrency: 5;
- cache/resume: enabled;
- run stage: `full`;
- budget label: `USER_APPROVED_LOCAL_FULLDATA_DEEPSEEK_20260501`.

Each input first passed `scripts/check_api_pilot_readiness.py`, then a
5-example dry-run, a 5-example real API smoke, and finally the full local gate
slice. All six full-stage runs completed with zero failed cases.

| Domain/gate | Count | Failed | GroundHit | Target correctness | ECE | WBC_tau | Generated in candidates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Health free-form gate60 | 60 | 0 | 0.117 | 0.000 | 0.762 | 0.967 | n/a |
| Health retrieval-context gate60 | 60 | 0 | 0.883 | 0.000 | 0.850 | 1.000 | 0.800 |
| Health catalog-constrained gate60 | 60 | 0 | 0.533 | 0.000 | 0.519 | 0.617 | 0.533 |
| Video_Games free-form gate30 | 30 | 0 | 0.300 | 0.000 | 0.745 | 0.900 | n/a |
| Video_Games retrieval-context gate30 | 30 | 0 | 0.900 | 0.000 | 0.794 | 0.933 | 0.800 |
| Video_Games catalog-constrained gate30 | 30 | 0 | 0.767 | 0.000 | 0.620 | 0.733 | 0.700 |

Ignored artifact roots:

```text
outputs/api_observations/deepseek/amazon_reviews_2023_health/full/test_gate60_no_repeat_*_api_full60_20260501/
outputs/api_observations/deepseek/amazon_reviews_2023_video_games/full/test_gate30_no_repeat_*_api_full30_20260501/
outputs/analysis/api_observations/deepseek/amazon_reviews_2023_health/full/test_gate60_no_repeat_*_api_full60_20260501/
outputs/analysis/api_observations/deepseek/amazon_reviews_2023_video_games/full/test_gate30_no_repeat_*_api_full30_20260501/
outputs/case_reviews/api_observations/deepseek/amazon_reviews_2023_health/full/test_gate60_no_repeat_*_api_full60_20260501/
outputs/case_reviews/api_observations/deepseek/amazon_reviews_2023_video_games/full/test_gate30_no_repeat_*_api_full30_20260501/
outputs/analysis_comparisons/deepseek/amazon_reviews_2023_health/full/gate60_prompt_shapes_20260501/
outputs/analysis_comparisons/deepseek/amazon_reviews_2023_video_games/full/gate30_prompt_shapes_20260501/
```

Interpretation guardrails:

- These are real DeepSeek API artifacts with manifests, not raw-response
  commits, not calibrated confidence, not trained TRUCE/CURE results, and not
  paper conclusions.
- Retrieval-context and catalog-constrained inputs exclude the held-out target,
  so target-hit correctness is not recommendation accuracy. Use these gates to
  diagnose grounding, candidate following, confidence behavior, and
  wrong-high-confidence risk before broader free-form or server-scale runs.
- The local Video_Games all-test input has 56,074 records and was not started
  in this local turn as a long-running paid API job. The closed artifact is the
  approved gate30 slice with dry-run, smoke, full run, analysis, case review,
  and comparison.
