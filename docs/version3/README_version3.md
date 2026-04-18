# Version 3 Guide

Version 3 upgrades the repository from a single legacy pointwise pipeline into a dual-line research framework:

- `pointwise_yesno` remains available as the protected legacy baseline
- `candidate_ranking` is the new ranking-oriented main line
- local Hugging Face backends are supported alongside API backends
- baseline-side confidence validation is isolated as an independent module
- ranking-aware uncertainty compare and ranking-aware rerank now have minimal runnable paths

This guide is the shortest way to understand what Version 3 added, what is already runnable, and which docs to open next.

## Part Index

- Part 1: legacy protection + local backend
  - [2026-04-18_part1_legacy_and_local_backend.md](./2026-04-18_part1_legacy_and_local_backend.md)
  - [2026-04-18_part1_backend_path_proof.md](./2026-04-18_part1_backend_path_proof.md)
- Part 2: candidate ranking minimal loop
  - [2026-04-18_part2_candidate_ranking_minimal_loop.md](./2026-04-18_part2_candidate_ranking_minimal_loop.md)
- Part 3: ranking eval minimal adapter
  - [2026-04-18_part3_ranking_eval_minimal_adapter.md](./2026-04-18_part3_ranking_eval_minimal_adapter.md)
- Part 4: baseline-side confidence validation
  - [2026-04-18_part4_baseline_confidence_validation.md](./2026-04-18_part4_baseline_confidence_validation.md)
- Part 5: ranking uncertainty and rerank adaptation
  - [2026-04-18_part5_ranking_uncertainty_and_rerank_adaptation.md](./2026-04-18_part5_ranking_uncertainty_and_rerank_adaptation.md)
- Part 6: Version 3 round-1 wrap-up
  - [2026-04-18_part6_version3_round1_summary.md](./2026-04-18_part6_version3_round1_summary.md)
  - [server_execution.md](./server_execution.md)

## Architecture Snapshot

Version 3 now has three clearly separated lines:

1. Legacy baseline line
   - API + `pointwise_yesno`
   - diagnosis / calibration / rerank / robustness
2. Ranking-oriented main line
   - `candidate_ranking`
   - ranking prompt -> ranking parser -> ranking eval
   - ranking proxy uncertainty compare -> ranking-aware rerank
3. Baseline-side confidence validation line
   - independent baseline score input
   - NH-style ranking metrics
   - confidence-like proxy diagnostics

## Recommended Smoke Matrix

These commands are the smallest reproducible checks for each Version 3 layer.

### A. Legacy pointwise yes/no smoke

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

## New Config Template

Version 3 now includes a tracked ranking-oriented local template:

- [beauty_qwen_local_rank.yaml](../../configs/exp/beauty_qwen_local_rank.yaml)

It is intended as the smallest server-first config example for:

- local backend
- `candidate_ranking`
- `score_list` prompting
- standard `outputs/{exp_name}/...` layout

## Output Layout Reminder

Version 3 does **not** introduce a new output tree. New lines still write under:

- `outputs/{exp_name}/predictions/`
- `outputs/{exp_name}/calibrated/`
- `outputs/{exp_name}/reranked/`
- `outputs/{exp_name}/figures/`
- `outputs/{exp_name}/tables/`

This applies to legacy pointwise, candidate ranking, and baseline-side confidence validation.

## What Part 6 Does Not Change

Part 6 is a wrap-up layer. It does not add:

- new backend logic
- new ranking prompt logic
- ranking-side calibration fitting
- new baseline training code
- new data builders

Its role is to make Version 3 easier to run, audit, and move to a server.
