# Summary Tables Guide

This document maps the generated summary CSV files under `outputs/summary/` to their intended use in paper writing. The goal is to keep the evidence layer stable: each table should answer a specific experimental question and should not drift into overlapping responsibilities.

## Core Tables

### `final_results.csv`

Use this as the primary cross-domain clean-results table.

Each row corresponds to one `(exp_name, domain, model, lambda)` setting and includes:

- diagnosis metrics
- test calibration metrics before/after calibration
- baseline ranking and exposure metrics
- uncertainty-aware rerank ranking and exposure metrics

Recommended use:

- main clean-result table
- cross-domain calibration discussion
- cross-domain reranking discussion

### `model_results.csv`

Use this as the main cross-model comparison table.

Each row corresponds to one `(domain, model, lambda)` experiment row and keeps:

- diagnosis metrics
- calibration metrics
- baseline ranking metrics
- rerank metrics

Recommended use:

- cross-model comparison on the clean pipeline
- model-centric analysis section

### `domain_model_summary.csv`

Grouped mean summary over `(domain, model)`.

Recommended use:

- compact appendix table
- quick sanity-check view over model behavior by domain

## Estimator Tables

### `estimator_results.csv`

Use this as the full multi-estimator comparison table.

Each row corresponds to one `(domain, model, estimator, lambda)` setting and includes:

- calibration metrics for the estimator-derived confidence signal
- baseline ranking/exposure metrics
- rerank metrics under the estimator-driven uncertainty

Recommended use:

- full estimator comparison appendix
- source table for custom filtered views

### `beauty_estimator_results.csv`

Beauty-only filtered estimator table.

Recommended use:

- main estimator-comparison table in the paper
- the central `Beauty x model x estimator` comparison view
- contains only the five primary clean Beauty experiment lines

### `beauty_estimator_supporting_results.csv`

Beauty-only supporting estimator table for non-main experiment variants.

Recommended use:

- appendix-level estimator support
- sensitivity or derived experiment bookkeeping without polluting the main Beauty estimator table

## Robustness Tables

### `robustness_results.csv`

Full clean-vs-noisy robustness summary.

Each row corresponds to one `(clean_exp, noisy_exp)` comparison and includes:

- ranking degradation
- calibration degradation
- high-confidence mistake changes

Recommended use:

- full robustness appendix
- detailed degradation inspection

### `robustness_brief.csv`

Compact robustness table.

Recommended use:

- main robustness table in the paper
- first clean/noisy claim

### `beauty_main_results.csv`

Beauty-only main-results table derived from `final_results.csv`.

Recommended use:

- main Beauty clean-results table
- first paper-facing table for the main domain

### `beauty_estimator_brief.csv`

Beauty-only estimator comparison brief table.

Recommended use:

- compact main-text estimator table
- quicker model-by-estimator comparison view

### `beauty_robustness_curve_brief.csv`

Beauty-only robustness curve table with noise-level organization.

Recommended use:

- main robustness curve table
- source table for a noise-level trend figure
- cross-model robustness support on Beauty (`deepseek` and `glm`)

### `beauty_reproducibility_brief.csv`

Beauty-only reproducibility brief table.

Recommended use:

- appendix stability table
- concise support for the reproducibility paragraph

### `beauty_consistency_sensitivity_brief.csv`

Beauty-only sampling-sensitivity brief table for `self-consistency`.

Recommended use:

- appendix table for consistency sensitivity
- source table for the claim that increasing temperature only partially activates consistency-based uncertainty

### `beauty_fused_alpha_brief.csv`

Beauty-only fusion-weight ablation brief table.

Recommended use:

- appendix table for `fused_alpha` analysis
- compact support for the claim that fused uncertainty is controllable but still weaker than calibrated verbalized confidence

## Auxiliary Tables

### `weekly_summary.csv`

Compact clean-result view with the most important diagnosis, calibration, and rerank metrics.

Recommended use:

- internal progress tracking
- quick reporting

### `rerank_ablation.csv`

Alias-style clean summary focused on domain/lambda organization.

Recommended use:

- lambda-focused reporting
- backward compatibility with earlier Week1/Week2 summaries

## Reproducibility Tables

### `reproducibility_check.csv`

Raw repeated-run metrics for the reproducibility check.

### `reproducibility_delta.csv`

Absolute metric differences between repeated runs.

Recommended use:

- reproducibility subsection
- appendix stability evidence

## Suggested Paper Mapping

- Table 1: `final_results.csv`
- Table 2: `model_results.csv`
- Main Beauty table: `beauty_main_results.csv`
- Main estimator table: `beauty_estimator_brief.csv`
- Main robustness table: `beauty_robustness_curve_brief.csv`
- Reproducibility appendix table: `beauty_reproducibility_brief.csv`
- Consistency-sensitivity appendix table: `beauty_consistency_sensitivity_brief.csv`
- Fused-alpha appendix table: `beauty_fused_alpha_brief.csv`

## Regeneration

To rebuild the current summary layer:

```powershell
py -3.12 main_aggregate_all.py --output_root outputs
```
