# Week7.8 Replay Manifest

## Purpose

This manifest records the Week7.8 local-v2 replay shell that replays the original teacher-requested uncertainty line under the current upgraded task structure.

## Replay Identity

- `week_stage`: `week7_8_replay`
- `model_source_group`: `local_hf_lora`
- `model_variant`: `srpd_v2_replay`
- `route_role`: `teacher_requested_local_mainline`

## Preserved Routes

- Official API observation line remains intact.
- Structured risk remains the strongest hand-crafted baseline.
- SRPD family remains a higher-layer trainable enhancement line and is not replaced by Week7.8.

## New Day1 Files

### Scope and bridge docs

- `docs/week7_8_local_v2_replay_scope.md`
- `docs/week7_8_replay_manifest.md`
- `docs/from_teacher_line_to_srpd_bridge.md`

### Replay model configs

- `configs/model/qwen3_8b_local_replay_v2_beauty_full973.yaml`
- `configs/model/qwen3_8b_local_replay_v2_books_full.yaml`
- `configs/model/qwen3_8b_local_replay_v2_electronics_full.yaml`
- `configs/model/qwen3_8b_local_replay_v2_movies_full.yaml`

### Replay exp skeletons

- pointwise:
  - `configs/exp/replay_v2_pointwise_beauty_full.yaml`
  - `configs/exp/replay_v2_pointwise_books_full.yaml`
  - `configs/exp/replay_v2_pointwise_electronics_full.yaml`
  - `configs/exp/replay_v2_pointwise_movies_full.yaml`
- ranking:
  - `configs/exp/replay_v2_rank_beauty_full973.yaml`
  - `configs/exp/replay_v2_rank_books_full.yaml`
  - `configs/exp/replay_v2_rank_electronics_full.yaml`
  - `configs/exp/replay_v2_rank_movies_full.yaml`
- rerank:
  - `configs/exp/replay_v2_rerank_beauty_full973.yaml`
  - `configs/exp/replay_v2_rerank_books_full.yaml`
  - `configs/exp/replay_v2_rerank_electronics_full.yaml`
  - `configs/exp/replay_v2_rerank_movies_full.yaml`

### Replay batch skeleton

- `configs/batch/week7_8_replay_v2_teacher_line.yaml`

## Readiness Status

### Beauty

- pointwise full: structurally ready
- ranking full973: structurally ready
- rerank full973: structurally ready

### Books

- full-domain data config exists
- replay configs added
- runtime files for `test.jsonl` and `ranking_test.jsonl` still need to be materialized under `data/processed/amazon_books/`

### Electronics

- full-domain data config exists
- replay configs added
- runtime files for `test.jsonl` and `ranking_test.jsonl` still need to be materialized under `data/processed/amazon_electronics/`

### Movies

- full-domain data config exists
- replay configs added
- runtime files for `test.jsonl` and `ranking_test.jsonl` still need to be materialized under `data/processed/amazon_movies/`

## Intended Output Families

- Week1-Week2 replay summary:
  `outputs/summary/week7_8_replay_v2_week1_week2_pointwise_summary.csv`
- Week3 replay compare:
  `outputs/summary/week7_8_replay_v2_week3_rerank_compare.csv`
- Week4 replay summary:
  `outputs/summary/week7_8_replay_v2_week4_robustness_summary.csv`
- Final teacher-requested local mainline summary:
  `outputs/summary/teacher_requested_local8b_lora_mainline_final.csv`

## Day1 Decision

Day1 does not attempt to force execution before the full-domain processed runtime files are aligned. The current goal is a clean, explicit, non-destructive replay shell.
