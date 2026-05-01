# Local DeepSeek Experiment Launch

This document defines the local experiment-entry layer for Storyflow /
TRUCE-Rec when DeepSeek API access and processed full datasets are available,
while Qwen3 server observation, LoRA training, and server-side baseline runs are
deferred.

The launcher writes a plan-only artifact. It does not call paid APIs, execute a
server, train a model, download data, or create paper evidence.

## Build A Local Plan

```powershell
python scripts/build_local_deepseek_experiment_plan.py
```

Default ignored output:

```text
outputs/experiment_plans/local_deepseek_fulldata_gate/
```

The plan covers the current local DeepSeek path:

- Amazon Reviews 2023 Beauty, Digital_Music, Handmade_Products,
  Health_and_Personal_Care, and Video_Games by default when local full
  processed artifacts exist;
- processed suffix `full`;
- test split;
- no-repeat target policy;
- forced-JSON, retrieval-context, and catalog-constrained prompt variants;
- rate limit `300` requests/minute and max concurrency `20`;
- gate size `60`;
- explicit server deferral.

## Standing Local Full-Domain Approval

As of 2026-05-01, the user approved the local DeepSeek execution default for
full-domain Amazon observations:

- provider/model: DeepSeek `deepseek-v4-flash`;
- budget: unlimited for the approved local full-domain DeepSeek run;
- speed policy: run fast, but keep experimental quality intact by preserving
  deterministic temperature, DeepSeek JSON output mode, bounded completion
  length, cache/resume, prompt templates, parsing, grounding, manifests, and
  downstream analysis;
- default budget label:
  `USER_APPROVED_UNLIMITED_FAST_QUALITY_DEEPSEEK_V4_FLASH_20260501`.

This standing approval removes the need to ask again for provider/model/budget
when executing this exact local DeepSeek full-domain path. It does not authorize
other paid providers, Qwen3/server inference, LoRA training, trained baseline
jobs, new raw-data downloads, or paper-result claims.

## 2026-05-01 Pause Update

The user paused the local Video_Games all-test DeepSeek run and decided that
the large-domain execution should be deferred to server or a later explicitly
started experiment phase. Until the user starts that phase, do not launch more
full-domain API jobs. Continue repository setup, plan generation, validation,
server runbook work, and readiness reporting only.

Adjust the scope without running an API:

```powershell
python scripts/build_local_deepseek_experiment_plan.py --dataset amazon_reviews_2023_beauty --dataset amazon_reviews_2023_health --gate-size 30 --run-stage pilot --rate-limit 30 --max-concurrency 3 --budget-label USER_APPROVED_LOCAL_DEEPSEEK_PILOT
```

## Execution Order

For each dataset, the generated report lists commands in this order:

1. Validate processed data.
2. Audit chronological split, repeat targets, title quality, and bucket
   coverage.
3. Build local observation-gate inputs.
4. Check API readiness without making a network call.
5. Run a dry-run observation without making a network call.
6. Run the `--execute-api` command directly for the approved DeepSeek
   `deepseek-v4-flash` full-domain path above; other execution paths still
   require explicit approval.
7. Analyze, case-review, and build CURE/TRUCE diagnostic features from the
   grounded predictions.

The generated `--execute-api` commands are authorized only for the exact
standing local DeepSeek path described above. They still require the API key
environment variable to be present, and any different provider/model/budget or
server/training path must be approved separately.

## Scope Guardrails

- The plan records `api_called=false`, `server_executed=false`,
  `model_training=false`, and `is_experiment_result=false`.
- DeepSeek API outputs become real observation artifacts only after the
  generated `--execute-api` command is explicitly run and its manifest is
  retained under ignored `outputs/`.
- Retrieval-context and catalog-constrained target-excluding gates diagnose
  prompt/candidate/grounding behavior; their target correctness must not be
  reported as recommendation accuracy.
- Same-split calibration/residualization commands are diagnostic only and must
  retain their leakage caveats.
- Qwen3-8B server observation, Qwen3-8B + LoRA training, and server-side trained
  baselines remain deferred until a separate server runbook step is approved.
