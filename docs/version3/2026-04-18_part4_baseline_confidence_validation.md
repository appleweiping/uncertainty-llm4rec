# 2026-04-18 Part 4: Baseline-Side Confidence Validation

## Stage Goal

Part 4 adds the Version 3 reviewer-defense line:

- keep the main research line untouched
- add an **independent** baseline-side confidence validation module
- read baseline candidate scores or rank rows
- compute NH-style performance metrics
- compute minimal confidence-like / uncertainty-like proxy diagnostics

This stage is intentionally **not** a giant baseline benchmark. It is a minimal-change external-validity module.

## Files Modified

Added:

- `src/baselines/__init__.py`
- `src/baselines/score_proxies.py`
- `main_baseline_confidence.py`
- `configs/baseline_confidence/minimal_score_rows.yaml`
- `docs/version3/examples/baseline_score_rows_example.jsonl`
- `docs/version3/2026-04-18_part4_baseline_confidence_validation.md`

Modified:

- `README.md` (tiny Version 3 note only)

## What Was Implemented

### 1. Independent baseline-side module

Part 4 does **not** touch:

- `main_infer.py`
- `main_eval.py`
- `main_rerank.py`
- `src/llm/*`
- `src/data/*`

Instead, baseline confidence validation is isolated behind:

- `src/baselines/score_proxies.py`
- `main_baseline_confidence.py`

### 2. Supported baseline input styles

The new module supports three minimal input styles:

- `score_rows`
- `rank_rows`
- `grouped_scores`

The default is `input_format: auto`, which infers the format from the columns.

The smoke-tested example uses `score_rows`, with one row per candidate:

```json
{"user_id":"u1","candidate_item_id":"i_pos_u1","score":3.2,"label":1,"target_popularity_group":"mid"}
```

### 3. NH-style performance outputs

The script builds a ranked dataframe and exports:

- `HR@K`
- `NDCG@K`
- `MRR@K`

Saved to:

- `outputs/{exp_name}/tables/baseline_ranking_metrics.csv`
- `outputs/{exp_name}/tables/baseline_ranking_rows.csv`

### 4. Confidence-like proxy diagnostics

For score-based baseline outputs, the script extracts:

- `top1_score`
- `top2_score`
- `score_margin`
- `proxy_confidence` (top-1 softmax probability)
- `score_entropy`
- `score_sharpness`
- `top1_accuracy`

It then exports:

- `baseline_proxy_summary.csv`
- `baseline_proxy_rows.csv`
- `baseline_proxy_bins_accuracy.csv`
- `baseline_proxy_reliability_bins.csv`
- `baseline_proxy_popularity_stats.csv`
- `baseline_margin_bins.csv`
- `baseline_proxy_grouped_rows.csv`

### 5. Minimal figures

Part 4 also writes lightweight paper/debug figures under the standard layout:

- `outputs/{exp_name}/figures/baseline_proxy_confidence_histogram.png`
- `outputs/{exp_name}/figures/baseline_proxy_reliability_diagram.png`
- `outputs/{exp_name}/figures/baseline_proxy_popularity_avg_confidence.png`

## What Was Explicitly Not Implemented

Part 4 does **not** include:

- ranking-side calibration
- ranking-side reranking
- uncertainty compare adaptation
- baseline model training code
- baseline robustness extension
- data-layer rewrites
- changes to the main research line

This stage is only the reviewer-defense baseline confidence line.

## Minimal Example Input

A tracked example file is included:

- `docs/version3/examples/baseline_score_rows_example.jsonl`

This example contains two users and three candidates per user, with one deliberately wrong top-1 baseline decision so that proxy diagnostics are non-trivial.

## Smoke Tests Completed

### A. Dump example helper

Command:

```powershell
py -3.12 main_baseline_confidence.py `
  --dump_example outputs/version3_part4_smoke/generated_example.jsonl
```

Confirmed:

- the helper writes a valid score-row JSONL example

### B. Baseline confidence smoke from config

Command:

```powershell
py -3.12 main_baseline_confidence.py `
  --config configs/baseline_confidence/minimal_score_rows.yaml
```

Confirmed:

- baseline input is read successfully
- `score_rows` is inferred / resolved correctly
- ranking dataframe is built
- NH metrics are exported
- proxy summary / bins / grouped tables are exported
- figures are written under the standard `outputs/{exp_name}/figures/` directory

### C. Equivalent explicit CLI smoke

Command:

```powershell
py -3.12 main_baseline_confidence.py `
  --exp_name version3_part4_baseline_smoke_cli `
  --input_path docs/version3/examples/baseline_score_rows_example.jsonl `
  --input_format score_rows `
  --output_root outputs `
  --k 3 `
  --n_bins 5 `
  --high_conf_threshold 0.8 `
  --seed 42
```

## Recommended Reproduction Commands

### 1. Smallest config-driven run

```powershell
py -3.12 main_baseline_confidence.py `
  --config configs/baseline_confidence/minimal_score_rows.yaml
```

### 2. Explicit CLI run

```powershell
py -3.12 main_baseline_confidence.py `
  --exp_name version3_part4_baseline_smoke_cli `
  --input_path docs/version3/examples/baseline_score_rows_example.jsonl `
  --input_format score_rows `
  --output_root outputs `
  --k 3 `
  --n_bins 5 `
  --high_conf_threshold 0.8 `
  --seed 42
```

## Acceptance Mapping

### 1. Is baseline-side confidence validation an independent module?

Yes.

The implementation is isolated to `src/baselines/score_proxies.py` and `main_baseline_confidence.py`.

### 2. Does it preserve the minimal-change principle?

Yes.

It consumes baseline scores or rank rows and extracts proxy confidence / uncertainty without modifying baseline training logic.

### 3. Does it output both NH performance and confidence-side diagnostics?

Yes.

It exports:

- NH metrics (`HR`, `NDCG`, `MRR`)
- proxy summary
- proxy bins / grouped rows
- proxy plots

### 4. Does it remain within the intended Part 4 boundary?

Yes.

This stage does not touch mainline inference, mainline evaluation, calibration, reranking, or uncertainty modules.
