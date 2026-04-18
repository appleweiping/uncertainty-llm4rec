# 2026-04-18 Part 5: Ranking Uncertainty and Rerank Adaptation

## Stage Goal

Part 5 starts reconnecting the ranking-oriented Version 3 line back to uncertainty and decision modules.

The goal in this stage is **not** to finish ranking-side calibration or a full ranking paper experiment. The goal is:

- keep the legacy pointwise uncertainty / rerank line intact
- extend the estimator registry so ranking proxy estimators become first-class entries
- let `main_uncertainty_compare.py` distinguish pointwise vs ranking/proxy inputs
- let `main_rerank.py` accept `candidate_ranking` inputs through a clean task-aware interface

This stage is a **structure upgrade**, not a full experimental completion.

## Files Modified

Modified:

- `src/uncertainty/estimators.py`
- `main_uncertainty_compare.py`
- `main_rerank.py`
- `README.md`

Added:

- `docs/version3/2026-04-18_part5_ranking_uncertainty_and_rerank_adaptation.md`

## What Was Implemented

### 1. Estimator registry now includes ranking-side proxy estimators

`src/uncertainty/estimators.py` is no longer limited to the pointwise family:

- `verbalized_raw`
- `verbalized_calibrated`
- `consistency`
- `fused`

It now also supports ranking-side proxy estimators:

- `score_margin`
- `score_entropy`

To support this, the module now provides:

- `build_ranking_proxy_dataframe(...)`
- `build_ranking_candidate_dataframe(...)`
- task-aware estimator column enrichment for ranking proxy rows

The ranking proxy line currently treats:

- `score_margin` as a confidence-like signal via a bounded margin transform
- `score_entropy` as an uncertainty-like signal derived from the score distribution sharpness

### 2. `main_uncertainty_compare.py` is now task-aware

`main_uncertainty_compare.py` now supports:

- `pointwise_yesno`
- `candidate_ranking`

Behavior:

- pointwise inputs continue through the old calibrated / consistency / fused compare path
- ranking inputs now build a ranking proxy dataframe from structured ranking predictions and compare:
  - `score_margin`
  - `score_entropy`

For ranking inputs, this stage focuses on:

- calibration-like proxy diagnostics
- carrying basic ranking context metrics into the comparison table

It does **not** yet attempt full ranking-side calibration or full ranking-side uncertainty experiments.

### 3. `main_rerank.py` now has a task-aware ranking branch

`main_rerank.py` now supports:

- `--task_type pointwise_yesno`
- `--task_type candidate_ranking`

Legacy pointwise behavior remains unchanged.

For `candidate_ranking`, the new branch:

- reads structured ranking prediction outputs
- expands them into candidate-level rows
- supports configurable ranking score / uncertainty sources

Current minimal ranking sources:

- `--ranking_score_source raw_score`
- `--ranking_score_source score_probability`

- `--ranking_uncertainty_source inverse_probability`
- `--ranking_uncertainty_source normalized_entropy`

This is intentionally a minimal interface-first adaptation. It proves that ranking-aware rerank can now be driven by task-aware score / uncertainty sources without rewriting the old pointwise path.

## What Was Explicitly Not Implemented

Part 5 does **not** include:

- ranking-side calibration fitting in `main_calibrate.py`
- ranking-side robustness
- baseline-side confidence changes
- backend or data-layer rewrites
- final ranking-oriented uncertainty research experiments

This stage only establishes the interface and minimal runnable path.

## Smoke Tests Completed

### A. Ranking uncertainty compare smoke

Command:

```powershell
py -3.12 main_uncertainty_compare.py `
  --exp_name version3_part5_ranking_compare_smoke `
  --task_type candidate_ranking `
  --input_path outputs/version3_part2_smoke/test_ranking_raw.jsonl `
  --output_root outputs `
  --k 5
```

Confirmed:

- ranking predictions are read successfully
- ranking proxy rows are built
- `score_margin` and `score_entropy` enter the estimator registry
- `estimator_comparison.csv` is exported

Key file:

- `outputs/version3_part5_ranking_compare_smoke/tables/estimator_comparison.csv`

### B. Ranking rerank smoke

Command:

```powershell
py -3.12 main_rerank.py `
  --exp_name version3_part5_ranking_rerank_smoke `
  --task_type candidate_ranking `
  --input_path outputs/version3_part2_smoke/test_ranking_raw.jsonl `
  --output_root outputs `
  --k 5 `
  --lambda_penalty 0.5 `
  --ranking_score_source raw_score `
  --ranking_uncertainty_source inverse_probability
```

Confirmed:

- ranking predictions are expanded to candidate-level rows
- baseline ranking output is written
- uncertainty-aware rerank output is written
- task-aware rerank results table is exported

Key files:

- `outputs/version3_part5_ranking_rerank_smoke/reranked/ranking_baseline_ranked.jsonl`
- `outputs/version3_part5_ranking_rerank_smoke/reranked/ranking_uncertainty_reranked.jsonl`
- `outputs/version3_part5_ranking_rerank_smoke/tables/rerank_results.csv`

### C. Legacy pointwise regression

Pointwise compare regression:

```powershell
py -3.12 main_uncertainty_compare.py `
  --exp_name version3_part5_pointwise_compare_regression `
  --calibrated_path outputs/beauty_deepseek/calibrated/test_calibrated.jsonl `
  --consistency_path outputs/beauty_deepseek/self_consistency/test_self_consistency.jsonl `
  --output_root outputs `
  --k 5 `
  --lambda_penalty 0.5
```

Pointwise rerank regression:

```powershell
py -3.12 main_rerank.py `
  --exp_name version3_part5_pointwise_rerank_regression `
  --task_type pointwise_yesno `
  --input_path outputs/beauty_deepseek/calibrated/test_calibrated.jsonl `
  --output_root outputs `
  --k 5 `
  --lambda_penalty 0.5 `
  --seed 42
```

Confirmed:

- legacy pointwise compare still runs
- legacy pointwise rerank still runs
- the new ranking branch did not break the old path

## Acceptance Mapping

### 1. Did the estimator layer become more like a unified registry?

Yes.

The registry now covers both:

- pointwise verbalized / calibrated / consistency / fused estimators
- ranking-side `score_margin` / `score_entropy` proxy estimators

### 2. Can `main_uncertainty_compare.py` distinguish pointwise and ranking settings?

Yes.

It now supports task-aware compare behavior rather than assuming every input is a pointwise confidence table.

### 3. Does `main_rerank.py` now reserve a clean ranking interface?

Yes.

The new `candidate_ranking` branch uses explicit task-aware score and uncertainty sources instead of hardcoding everything into the legacy pointwise path.

### 4. Is this still within the intended Part 5 boundary?

Yes.

This stage adapts:

- estimators
- uncertainty compare
- rerank

without reopening backend, data, or baseline modules.
