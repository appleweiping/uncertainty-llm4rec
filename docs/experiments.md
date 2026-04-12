# Experiments Guide

This document maps the current repository structure to the main experiment families used in the project. It is meant to serve as the practical companion to the paper outline: each experiment family below points to the relevant entry scripts, config layers, and summary files.

## Config Layers

The repository is organized around three config layers:

- `configs/data/`: data preprocessing and sample-building settings
- `configs/model/`: backend, model, connection, and generation settings
- `configs/exp/`: concrete experiment runs with `exp_name`, `input_path`, `model_config`, and output root

In practice:

- switch domain by changing `configs/data/*.yaml`
- switch model by changing `configs/model/*.yaml`
- switch a concrete run by changing `configs/exp/*.yaml`

## Main Experiment Families

### 1. Clean Diagnosis / Calibration / Reranking

This is the main Week1 pipeline:

```text
main_preprocess.py
-> main_build_samples.py
-> main_infer.py
-> main_eval.py
-> main_calibrate.py
-> main_rerank.py
```

Representative experiment names:

- `beauty_deepseek`
- `beauty_qwen`
- `beauty_glm`
- `beauty_kimi`
- `beauty_doubao`

Cross-domain small-subset validation:

- `movies_small_*`
- `books_small_*`
- `electronics_small_*`

Primary outputs:

- `outputs/{exp_name}/predictions/`
- `outputs/{exp_name}/calibrated/`
- `outputs/{exp_name}/reranked/`
- `outputs/{exp_name}/tables/`

### 2. Multi-Estimator Comparison

This is the Week2 Day3-Day4 line:

```text
main_self_consistency.py
-> main_uncertainty_compare.py
-> src/analysis/aggregate_estimator_results.py
```

Current estimator types:

- `verbalized_raw`
- `verbalized_calibrated`
- `consistency`
- `fused`

Key summary outputs:

- `outputs/summary/estimator_results.csv`
- `outputs/summary/beauty_estimator_results.csv`

### 3. Robustness / Noisy

This is the Week2 Day5 line:

```text
main_generate_noisy.py
-> main_infer.py
-> main_eval.py
-> main_calibrate.py
-> main_rerank.py
-> main_robustness.py
```

Current reference setting:

- `beauty_deepseek_noisy`

Key robustness outputs:

- `outputs/robustness/{clean_exp}_vs_{noisy_exp}/tables/robustness_table.csv`
- `outputs/robustness/{clean_exp}_vs_{noisy_exp}/tables/robustness_calibration_table.csv`
- `outputs/robustness/{clean_exp}_vs_{noisy_exp}/tables/robustness_confidence_table.csv`
- `outputs/summary/robustness_results.csv`
- `outputs/summary/robustness_brief.csv`

### 4. Experiment Aggregation

This is the Week2 Day6 line:

```text
main_aggregate_all.py
```

It calls:

- `src/analysis/aggregate_domain_results.py`
- `src/analysis/aggregate_model_results.py`
- `src/analysis/aggregate_estimator_results.py`
- `src/analysis/robustness_summary.py`

Run:

```powershell
py -3.12 main_aggregate_all.py --output_root outputs
```

Generated summary files under `outputs/summary/`:

- `final_results.csv`
- `weekly_summary.csv`
- `rerank_ablation.csv`
- `model_results.csv`
- `domain_model_summary.csv`
- `estimator_results.csv`
- `beauty_estimator_results.csv`
- `robustness_results.csv`
- `robustness_brief.csv`
- `beauty_main_results.csv`
- `beauty_estimator_brief.csv`
- `beauty_robustness_curve_brief.csv`
- `beauty_reproducibility_brief.csv`

For table-level responsibilities and paper mapping, see:

- [docs/tables.md](tables.md)
- [docs/paper_outline.md](paper_outline.md)
- [docs/beauty_freeze_checklist.md](beauty_freeze_checklist.md)

## Suggested Reading Order for Paper Writing

If you are mapping code/results to the paper, the cleanest order is:

1. `outputs/summary/final_results.csv`
   Use for cross-domain and cross-model diagnosis/calibration/rerank discussion.
2. `outputs/summary/beauty_estimator_results.csv`
   Use for the main multi-estimator comparison table.
3. `outputs/summary/robustness_brief.csv`
   Use for the first clean/noisy robustness claim.
4. `outputs/summary/reproducibility_delta.csv`
   Use for the reproducibility appendix or stability note.

For Beauty-first paper writing, the most direct files are:

- `outputs/summary/beauty_main_results.csv`
- `outputs/summary/beauty_estimator_brief.csv`
- `outputs/summary/beauty_robustness_curve_brief.csv`
- `outputs/summary/beauty_reproducibility_brief.csv`

## Current Practical Baselines

At the current stage, the most stable reference settings are:

- `beauty_deepseek` for the main clean baseline
- `beauty_deepseek_noisy` for the first robustness baseline
- `beauty_*` for the full five-model estimator comparison
- `movies_small_deepseek` for the minimal cross-domain estimator validation

## Notes

- Current `100`-sample runs are best treated as stable research baselines, not final large-scale paper numbers.
- The summary layer is designed so later larger runs can replace or extend existing rows without changing file formats.
- `Paper/` local reports are intentionally not part of the reproducibility surface; the reproducibility surface is the config + script + summary stack.
