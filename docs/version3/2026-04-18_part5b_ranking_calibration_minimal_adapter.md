# 2026-04-18 Part 5b: Ranking-Side Calibration Minimal Adapter

## Stage Goal

This follow-up stage closes the most obvious Version 3 gap after Parts 1 to 6:

- `candidate_ranking` already had infer
- `candidate_ranking` already had eval
- `candidate_ranking` already had proxy uncertainty compare
- `candidate_ranking` already had task-aware rerank

But `main_calibrate.py` was still pointwise-only.

The goal of this stage is therefore:

- let `main_calibrate.py` recognize `candidate_ranking`
- reuse the existing valid-fit / test-apply calibration protocol
- calibrate a ranking-side proxy confidence signal
- emit a minimal calibrated output schema without breaking legacy pointwise calibration

This stage does **not** attempt to complete full ranking research migration. It only closes the calibration entry-point gap.

## Files Modified

Modified:

- `main_calibrate.py`
- `README.md`
- `docs/version3/README_version3.md`

Added:

- `docs/version3/2026-04-18_part5b_ranking_calibration_minimal_adapter.md`

## What Was Implemented

### 1. `main_calibrate.py` now recognizes `candidate_ranking`

Added:

- `--task_type auto | pointwise_yesno | candidate_ranking`

and a small task-type inference helper.

The calibration entry now supports:

- legacy pointwise raw predictions
- ranking raw predictions

without removing the old pointwise path.

### 2. Ranking raw predictions are converted into calibration-ready proxy rows

For `candidate_ranking`, the code now:

1. loads ranking raw prediction JSONL
2. builds proxy rows through `build_ranking_proxy_dataframe(...)`
3. uses `proxy_confidence` as the ranking-side confidence signal
4. aliases it into `confidence` so the existing calibration metrics stack can be reused

This preserves the current Version 3 minimal-change philosophy:

- no new ranking-specific calibrator class
- no new training logic
- reuse the existing valid-fit / test-apply calibration machinery

### 3. Ranking-side calibrated output schema

The calibrated ranking proxy rows now include:

- `proxy_confidence`
- `confidence`
- `calibrated_confidence`
- `calibrated_proxy_confidence`
- `uncertainty`

alongside the existing ranking proxy fields such as:

- `user_id`
- `target_item_id`
- `selected_item_id`
- `top1_score`
- `top2_score`
- `score_margin`
- `score_entropy`
- `target_popularity_group`

### 4. Output files

For ranking-side calibration, standard outputs still go under:

- `outputs/{exp_name}/calibrated/`
- `outputs/{exp_name}/tables/`
- `outputs/{exp_name}/figures/`

New ranking-oriented table exports include:

- `tables/ranking_proxy_valid_calibrated.csv`
- `tables/ranking_proxy_test_calibrated.csv`

while the shared outputs remain:

- `calibrated/valid_calibrated.jsonl`
- `calibrated/test_calibrated.jsonl`
- `tables/calibration_comparison.csv`
- `tables/calibration_split_metadata.csv`
- `tables/reliability_before_calibration.csv`
- `tables/reliability_after_calibration.csv`

## Smoke Tests Completed

### A. Legacy pointwise calibration regression

Command:

```powershell
py -3.12 main_calibrate.py `
  --exp_name version3_calibrate_pointwise_smoke `
  --valid_path outputs/beauty_deepseek/predictions/valid_raw.jsonl `
  --test_path outputs/beauty_deepseek/predictions/test_raw.jsonl `
  --output_root outputs `
  --method isotonic `
  --task_type pointwise_yesno
```

Result:

- passed
- pointwise calibration path remains usable
- legacy output schema remains intact

### B. Ranking-side calibration smoke

The existing `outputs/version3_part2_smoke/test_ranking_raw.jsonl` only contained one user, so direct user-level split correctly failed with:

- `ValueError: Need at least 2 unique users for valid/test split.`

That failure is useful because it proves the new ranking path really reached:

- ranking proxy build
- user-level split
- calibration entry logic

To complete a runnable smoke, a tiny temporary two-user ranking input was generated from the same raw row.

Command:

```powershell
py -3.12 main_calibrate.py `
  --exp_name version3_calibrate_ranking_smoke `
  --input_path outputs/version3_calibrate_ranking_smoke_input/two_user_ranking_raw.jsonl `
  --output_root outputs `
  --method isotonic `
  --task_type candidate_ranking
```

Result:

- passed
- ranking-side calibration path is now runnable
- ranking proxy calibrated tables and JSONL outputs were written successfully

## Current Limits

This stage still does **not** complete the full Version 3 calibration story.

What is still missing:

- calibration integrated into a full multi-model ranking experiment matrix
- calibration integrated into multi-domain ranking experiments
- stronger ranking-side confidence families beyond the current proxy confidence route
- ranking-side calibration under robustness settings

So the correct reading is:

- ranking calibration entry-point gap: closed
- full ranking calibration research layer: not yet complete

## Recommended Next Step

The next most valuable follow-up is still:

1. choose real NH baseline papers
2. connect 2 to 3 selected baselines to the existing baseline-confidence framework
3. then begin multi-model / multi-domain migration on the ranking main line
