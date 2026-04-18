# 2026-04-18 Part 6: Version 3 Round-1 Summary

## Stage Goal

Part 6 is the wrap-up stage for the first full Version 3 pass.

It does **not** add new model logic or new research modules. Its goal is to make the repository:

- easier to hand off
- easier to run on a server
- easier to audit by part
- easier to reuse for formal experiments after local smoke tests

## Files Modified

Modified:

- `README.md`

Added:

- `configs/exp/beauty_qwen_local_rank.yaml`
- `docs/version3/README_version3.md`
- `docs/version3/server_execution.md`
- `docs/version3/2026-04-18_part6_version3_round1_summary.md`

## What Was Implemented

### 1. Version 3 top-level doc index

Added:

- `docs/version3/README_version3.md`

This file now acts as the Version 3 landing page and explains:

- the dual-line structure
- which part added which capability
- the smoke matrix across Parts 1 to 5
- the new tracked ranking config template
- the shared `outputs/{exp_name}/...` layout

### 2. Server-first formal run guide

Added:

- `docs/version3/server_execution.md`

This doc provides copy-paste oriented commands for:

- environment creation
- optional API environment variables
- local Hugging Face model preparation
- legacy yes/no baseline execution
- local ranking main-line execution
- baseline-side confidence validation
- ranking uncertainty compare
- ranking-aware rerank

### 3. New tracked Version 3 config template

Added:

- `configs/exp/beauty_qwen_local_rank.yaml`

This config is the first tracked server-first example that combines:

- local model backend
- `candidate_ranking`
- `score_list` ranking mode
- standard output directory management

### 4. README touch-up

`README.md` now points readers to the new Version 3 docs rather than leaving the rollout logic spread across stage reports only.

## Full Smoke Command Summary

### A. Legacy pointwise smoke

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek.yaml `
  --input_path data/processed/amazon_beauty/valid.jsonl `
  --output_path outputs/version3_part6_pointwise_smoke/predictions/valid_raw.jsonl `
  --split_name valid `
  --max_samples 1 `
  --overwrite
```

### B. Local backend smoke

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek.yaml `
  --model_config configs/model/qwen_local_7b.yaml `
  --input_path data/processed/amazon_beauty/valid.jsonl `
  --output_path outputs/version3_part6_local_backend_smoke/predictions/valid_raw.jsonl `
  --split_name valid `
  --max_samples 1 `
  --overwrite
```

### C. Candidate ranking smoke

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_qwen_local_rank.yaml `
  --max_samples 5 `
  --overwrite
```

### D. Ranking eval smoke

```powershell
py -3.12 main_eval.py `
  --exp_name version3_part3_ranking_smoke `
  --input_path outputs/version3_part2_smoke/test_ranking_raw.jsonl `
  --task_type candidate_ranking `
  --output_root outputs `
  --seed 42 `
  --k 5
```

### E. Baseline-side confidence smoke

```powershell
py -3.12 main_baseline_confidence.py `
  --config configs/baseline_confidence/minimal_score_rows.yaml
```

### F. Ranking uncertainty compare smoke

```powershell
py -3.12 main_uncertainty_compare.py `
  --exp_name version3_part5_ranking_compare_smoke `
  --task_type candidate_ranking `
  --input_path outputs/version3_part2_smoke/test_ranking_raw.jsonl `
  --output_root outputs `
  --k 5
```

### G. Ranking-aware rerank smoke

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

## What Part 6 Explicitly Does Not Do

Part 6 does **not**:

- change local backend behavior
- change prompt or parser behavior
- add ranking-side calibration fitting
- add new baseline model wrappers
- change the data layer
- reopen Parts 1 to 5

It is strictly a documentation, handoff, and run-template stage.

## Acceptance Mapping

### 1. Are the two main lines now explicitly documented?

Yes.

The docs now clearly separate:

- legacy `pointwise_yesno`
- ranking-oriented `candidate_ranking`

### 2. Is there a server-ready run guide?

Yes.

`docs/version3/server_execution.md` is designed to be copied onto a server workflow directly.

### 3. Is there at least one tracked new exp config template?

Yes.

`configs/exp/beauty_qwen_local_rank.yaml` provides that template.

### 4. Does Part 6 stay within its intended boundary?

Yes.

This stage only adds docs and run templates, plus one small tracked config example.
