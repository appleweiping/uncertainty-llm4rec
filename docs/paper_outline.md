# Paper Outline

This document turns the current code and summary stack into a writing plan for the Beauty-led main paper. The working principle is simple: every major claim in the paper should point to a stable summary table rather than an ad hoc experiment folder.

## Core Story

The paper should present one coherent evidence chain:

1. LLM verbalized confidence is informative but miscalibrated in recommendation.
2. Post-hoc calibration turns this raw signal into a more decision-ready probability proxy.
3. Calibrated uncertainty can be used at ranking time without degrading the clean pipeline.
4. Uncertainty is not limited to one source; verbalized and consistency-based signals can be compared and lightly fused.
5. Under input perturbation, uncertainty-aware behavior can be analyzed as a robustness question rather than only a clean-ranking question.
6. The full result stack is reproducible and structurally organized.

Beauty is the main experimental domain for the narrative above. Other domains are supporting evidence, not the writing center of gravity.

## Section Plan

### 1. Introduction

Main questions:

- Can LLM confidence be trusted in recommendation?
- If not, can it be corrected and operationalized?
- Does uncertainty-awareness remain meaningful once we move from analysis to decision making and robustness?

Claims supported by current results:

- raw verbalized confidence is miscalibrated
- calibration materially reduces ECE and Brier score
- uncertainty can be moved into reranking and robustness analysis
- the phenomenon is not tied to one model family

### 2. Method

Recommended structure:

1. Pointwise recommendation setup
2. LLM inference with verbalized confidence
3. Diagnosis metrics
4. Post-hoc calibration
5. Uncertainty-aware reranking
6. Multi-source uncertainty
7. Robustness under lightweight perturbations

Method notes:

- keep the method section centered on interpretable choices
- emphasize leakage-aware calibration: fit on `valid`, apply on `test`
- present fused uncertainty as a lightweight extension, not the headline novelty

### 3. Main Clean Results

Primary table source:

- `outputs/summary/beauty_main_results.csv`

Use this section to establish:

- clean pipeline quality on Beauty
- calibration improvement after post-hoc mapping
- reranking behavior and exposure structure

### 4. Cross-Model Comparison

Primary table source:

- `outputs/summary/model_results.csv`

Beauty-focused reading:

- filter `domain == beauty`

Use this section to argue:

- the confidence/calibration phenomenon is not a DeepSeek-only artifact
- five model families show a common diagnosis-to-calibration pattern

### 5. Multi-Estimator Comparison

Primary table sources:

- `outputs/summary/beauty_estimator_results.csv`
- `outputs/summary/beauty_estimator_brief.csv`

Core claims:

- calibrated verbalized confidence is the strongest stable estimator under the current setting
- consistency is meaningful as a second uncertainty source, but can degenerate under deterministic or low-variance sampling
- simple fusion is feasible and interpretable, even if it does not dominate calibrated verbalized confidence

### 6. Robustness

Primary table sources:

- `outputs/summary/robustness_brief.csv`
- `outputs/summary/beauty_robustness_curve_brief.csv`

Core claims:

- the project now has a clean-to-noisy robustness pipeline
- robustness should be discussed as a curve, not a single noisy point
- finer ranking metrics degrade earlier than coarse hit-rate metrics as noise increases

### 7. Reproducibility

Primary table sources:

- `outputs/summary/reproducibility_check.csv`
- `outputs/summary/reproducibility_delta.csv`
- `outputs/summary/beauty_reproducibility_brief.csv`

Core claims:

- repeated runs under fixed seed stay stable on main ranking and exposure metrics
- the main residual variance is concentrated in calibration bucket statistics on small evaluation sets

### 8. Discussion and Limitations

Recommended topics:

- current `100`-sample runs are stable research baselines, not final large-scale numbers
- consistency uncertainty is highly sensitive to decoding setup
- robustness conclusions currently come from Beauty-first evidence
- cross-domain support exists, but Beauty remains the main writing domain

## Table Mapping

- Table 1: `outputs/summary/beauty_main_results.csv`
- Table 2: `outputs/summary/model_results.csv` filtered to `domain=beauty`
- Table 3: `outputs/summary/beauty_estimator_brief.csv`
- Table 4: `outputs/summary/beauty_robustness_curve_brief.csv`
- Appendix Table A1: `outputs/summary/beauty_reproducibility_brief.csv`

## Writing Guidance

- Keep Beauty as the narrative anchor.
- Use other domains only to support generality claims.
- Do not let the paper drift into “many experiments” without a clear evidence chain.
- Every paragraph in the experiment section should be traceable to one stable CSV in `outputs/summary/`.
