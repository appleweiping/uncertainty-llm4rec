# 2026-04-18 Part 3: Ranking Eval Minimal Adapter

## Stage Goal

Part 3 completes the next missing piece of the Version 3 ranking line:

- keep legacy pointwise evaluation available
- let `main_eval.py` read `candidate_ranking` prediction files
- compute minimal NH-style ranking metrics
- export a lightweight confidence-like proxy summary for ranking outputs

This stage is still intentionally minimal. It is **not** the ranking-side calibration, reranking, uncertainty-comparison, or baseline-side confidence stage.

## Files Modified

Modified:

- `main_eval.py`
- `README.md`

Added:

- `docs/version3/2026-04-18_part3_ranking_eval_minimal_adapter.md`

## What Was Implemented

### 1. Task-aware evaluation in `main_eval.py`

`main_eval.py` now supports:

- `--task_type auto`
- `--task_type pointwise_yesno`
- `--task_type candidate_ranking`

Behavior:

- pointwise predictions still go through the old confidence diagnostics path
- ranking predictions now go through a separate ranking evaluation branch

### 2. Minimal ranking prediction reader

The ranking branch reads structured JSONL rows produced by Part 2 and extracts:

- `user_id`
- `target_item_id`
- `target_popularity_group`
- `ranked_item_ids`
- `candidate_scores`
- `selected_item_id`
- `candidates`

It then expands these rows into a ranking dataframe with:

- `user_id`
- `candidate_item_id`
- `label`
- `rank`
- `score`
- `target_popularity_group`

### 3. NH-style ranking metrics

The ranking branch now computes minimal NH-style metrics through the existing `src/eval/ranking_metrics.py`:

- `HR@K`
- `NDCG@K`
- `MRR@K`

These are saved to:

- `outputs/{exp_name}/tables/ranking_metrics.csv`

### 4. Minimal ranking confidence-like proxy summary

For `score_list`-style outputs, `main_eval.py` now also computes a minimal proxy summary:

- `top1_accuracy`
- `avg_top1_score`
- `avg_top2_score`
- `avg_score_margin`
- `avg_score_entropy`

These are saved to:

- `outputs/{exp_name}/tables/ranking_proxy_summary.csv`
- `outputs/{exp_name}/tables/ranking_proxy_rows.csv`

This gives the ranking line a first confidence-like diagnostic surface without yet entering calibration or uncertainty comparison.

## What Was Explicitly Not Implemented

Part 3 does **not** include:

- ranking-side calibration
- ranking-side reranking
- `main_uncertainty_compare.py` adaptation
- `src/uncertainty/*` changes
- baseline-side confidence validation
- data-layer rewrites
- large-scale ranking experiments

Those are deferred to later Version 3 parts.

## Smoke Tests Completed

### A. Legacy pointwise eval regression

Command used:

```powershell
py -3.12 main_eval.py `
  --exp_name version3_part3_pointwise_smoke `
  --input_path outputs/version3_part2_smoke/test_pointwise_raw.jsonl `
  --output_root outputs `
  --seed 42
```

Confirmed:

- pointwise task is still correctly detected / handled
- legacy confidence diagnostics still run
- tables and figures are saved under the standard `outputs/{exp_name}/...` layout

### B. Ranking eval smoke

Command used:

```powershell
py -3.12 main_eval.py `
  --exp_name version3_part3_ranking_smoke `
  --input_path outputs/version3_part2_smoke/test_ranking_raw.jsonl `
  --task_type candidate_ranking `
  --output_root outputs `
  --seed 42 `
  --k 5
```

Confirmed:

- ranking prediction file is read successfully
- ranking rows are expanded successfully
- minimal NH metrics are computed
- proxy summary is exported

Key output files:

- `outputs/version3_part3_ranking_smoke/tables/ranking_metrics.csv`
- `outputs/version3_part3_ranking_smoke/tables/ranking_proxy_summary.csv`
- `outputs/version3_part3_ranking_smoke/tables/ranking_rows.csv`

## Recommended Reproduction Commands

### 1. Legacy pointwise eval

```powershell
py -3.12 main_eval.py `
  --exp_name beauty_deepseek_eval_smoke `
  --input_path outputs/version3_part2_smoke/test_pointwise_raw.jsonl `
  --output_root outputs `
  --seed 42
```

### 2. Ranking eval with `score_list`

```powershell
py -3.12 main_eval.py `
  --exp_name beauty_deepseek_rank_eval_smoke `
  --input_path outputs/version3_part2_smoke/test_ranking_raw.jsonl `
  --task_type candidate_ranking `
  --output_root outputs `
  --seed 42 `
  --k 5
```

### 3. Ranking eval with `auto` task detection

```powershell
py -3.12 main_eval.py `
  --exp_name beauty_deepseek_rank_eval_auto `
  --input_path outputs/version3_part2_smoke/test_ranking_raw.jsonl `
  --task_type auto `
  --output_root outputs `
  --seed 42 `
  --k 5
```

## Acceptance Mapping

### 1. Does legacy pointwise eval still work?

Yes.

The old pointwise diagnostics path remains intact.

### 2. Can `main_eval.py` now read ranking outputs?

Yes.

It reads structured ranking JSONL and expands it into a ranking dataframe.

### 3. Does the ranking branch produce actual evaluation outputs?

Yes.

At minimum it now exports:

- ranking metrics
- ranking rows
- ranking proxy summary

### 4. Is this still within the intended Day 3 boundary?

Yes.

This is still a minimal adapter stage:

- read ranking predictions
- compute basic NH metrics
- expose a lightweight score-proxy diagnostic

It does not yet adapt calibration, reranking, uncertainty compare, or baseline confidence validation.
