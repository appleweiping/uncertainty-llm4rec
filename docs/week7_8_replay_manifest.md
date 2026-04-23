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

## New Day2 Files

### Runtime materialization

- `main_build_replay_runtime.py`

This script materializes the current full-domain replay runtime shape directly under each processed directory:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`
- `ranking_valid.jsonl`
- `ranking_test.jsonl`

### Pointwise replay summary

- `main_compare_teacher_requested_line.py`
- `configs/batch/week7_8_replay_v2_day2_pointwise.yaml`

This summary entrypoint is scoped to the Week7.8 Day2 replay target: local-v2 pointwise diagnosis and calibration on the formal four-domain route, together with the preserved historical teacher-observation reference.

## New Day3 Files

### Ranking and rerank batch shell

- `configs/batch/week7_8_replay_v2_day3_ranking_rerank.yaml`

This batch shell separates the Day3 decision layer from Day2 pointwise replay and makes the dependency order explicit: replay direct ranking depends on runtime-aligned ranking inputs, and replay structured-risk rerank depends on both replay pointwise calibration and replay ranking output.

### Ranking/rerank compare summary

- `main_compare_teacher_requested_line.py --mode rerank`

Day3 reuses the same teacher-requested summary entrypoint but switches to rerank mode, producing:

- `outputs/summary/week7_8_replay_v2_week3_rerank_compare.csv`
- `outputs/summary/week7_8_replay_v2_week3_rerank_compare.md`

The Day3 compare table is intentionally structured around three roles:

- replay direct local-v2 ranking
- replay local-v2 structured-risk rerank
- strongest preserved hand-crafted reference

## Readiness Status

### Beauty

- pointwise full: structurally ready
- ranking full973: structurally ready
- rerank full973: structurally ready

### Books

- full-domain data config exists
- replay configs added
- Day2 runtime builder now exists
- full-domain pointwise/ranking runtime files can now be materialized under `data/processed/amazon_books/`
- full-domain replay-v2 adapter is still not yet materialized

### Electronics

- full-domain data config exists
- replay configs added
- Day2 runtime builder now exists
- full-domain pointwise/ranking runtime files can now be materialized under `data/processed/amazon_electronics/`
- full-domain replay-v2 adapter is still not yet materialized

### Movies

- full-domain data config exists
- replay configs added
- Day2 runtime builder now exists
- full-domain pointwise/ranking runtime files can now be materialized under `data/processed/amazon_movies/`
- full-domain replay-v2 adapter is still not yet materialized

## Intended Output Families

- Week1-Week2 replay summary:
  `outputs/summary/week7_8_replay_v2_week1_week2_pointwise_summary.csv`
- Week3 replay compare:
  `outputs/summary/week7_8_replay_v2_week3_rerank_compare.csv`
- Week4 replay summary:
  `outputs/summary/week7_8_replay_v2_week4_robustness_summary.csv`
- Final teacher-requested local mainline summary:
  `outputs/summary/teacher_requested_local8b_lora_mainline_final.csv`

## Day2-Day3 Decision

Day2 moves the main blocker one layer lower. The processed full-domain directories can now be aligned to the runtime file shape used by the current pointwise and candidate-ranking entrypoints, and the formal pointwise/calibration summary entrypoint also exists.

Day3 then formalizes the decision-layer compare shell. Even before all replay runs are materialized, the compare table can already show a stable three-column structure across domains: replay direct, replay rerank, and strongest preserved hand-crafted reference. The remaining execution blocker is no longer compare structure, but full-domain replay-v2 adapter materialization for Books, Electronics, and Movies, together with the actual server-side replay outputs.
