# Local DeepSeek Experiment Launch

This document defines the local experiment-entry layer for Storyflow /
TRUCE-Rec when DeepSeek API access and processed full datasets are available,
while Qwen3 server observation, LoRA training, and server-side baseline runs are
deferred.

The launcher writes a plan-only artifact. It does not call paid APIs, execute a
server, train a model, download data, or create paper evidence.

## Build A Local Plan

```powershell
python scripts/build_local_deepseek_experiment_plan.py --budget-label USER_APPROVED_LOCAL_DEEPSEEK_EXPERIMENT
```

Default ignored output:

```text
outputs/experiment_plans/local_deepseek_fulldata_gate/
```

The plan covers the current local DeepSeek path:

- Amazon Reviews 2023 Beauty, Health, and Video Games by default;
- processed suffix `full`;
- test split;
- no-repeat target policy;
- forced-JSON, retrieval-context, and catalog-constrained prompt variants;
- rate limit `60` requests/minute and max concurrency `5`;
- gate size `60`;
- explicit server deferral.

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
6. Run the `--execute-api` command only after explicit approval in the current
   task.
7. Analyze, case-review, and build CURE/TRUCE diagnostic features from the
   grounded predictions.

The generated `--execute-api` commands are not authorized by the plan itself.
They still require the current task to approve provider, model, budget label,
sample size, rate limit, concurrency, and API key environment availability.

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
