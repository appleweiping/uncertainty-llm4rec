# Uncertainty-Aware LLMs for Recommendation
### Calibration, Bias, and Robust Decision Making

> A systematic research framework for analyzing, calibrating, and leveraging uncertainty in LLM-based recommendation systems.

---

## Overview

Large language models are increasingly applied to recommendation tasks, where they reason over structured user-item interactions and produce natural language judgments. Unlike traditional recommender systems, LLMs typically lack reliable uncertainty estimates — their outputs come with no inherent notion of how confident the model should be.

This project addresses a central question: **Can LLM-expressed confidence be trusted as a decision signal in recommendation?** And if not, how should it be corrected — and ultimately used to improve recommendations?

The answer, as this project demonstrates, is nuanced. LLM confidence is not random noise. It carries real discriminative information. But it is also systematically miscalibrated, structurally biased, and requires correction before it can responsibly inform downstream decisions. This project builds a complete research pipeline to study, fix, and exploit that signal.

---

## Motivation

When an LLM recommends an item with "90% confidence," what does that number actually mean? In most deployments, very little — because LLM verbalized confidence is:

- **Miscalibrated**: The stated confidence does not correspond to actual prediction accuracy
- **Structurally biased**: Confidence correlates with item popularity, creating systematic disparities across exposure groups
- **Underutilized**: Even when informative, confidence is rarely fed back into the ranking or decision process

This project treats LLM confidence not as an afterthought, but as a first-class variable deserving systematic study. We ask: how should uncertainty be defined, measured, corrected, and integrated into recommendation decisions?

---

## Method Pipeline

The project follows a complete, end-to-end research chain:

```
Pointwise LLM Inference
        ↓
Confidence Diagnosis & Calibration Analysis
        ↓
Post-Hoc Calibration (Platt / Isotonic)
        ↓
Uncertainty-Aware Reranking
        ↓
Multi-Source Uncertainty Modeling
        ↓
Robustness Evaluation under Input Perturbation
```

Each stage builds directly on the outputs of the previous one, forming a coherent evidence chain from problem identification to system-level validation.

---

## Experiment Stages

### Stage 1 — Pointwise Inference Loop

Built a minimal, reproducible LLM recommendation pipeline. Given structured user-item samples, the system automatically constructs prompts, calls the LLM, parses JSON-style outputs, and produces structured prediction files containing `recommend`, `confidence`, `reason`, and the raw model response. This stage establishes the experimental substrate for all downstream analysis.

### Stage 2 — Confidence Diagnosis

Conducted systematic diagnostic analysis of raw LLM confidence:

- **Discrimination analysis**: Relationship between confidence and correctness; prediction behavior at high vs. low confidence
- **Calibration quality**: Reliability diagrams, ECE, Brier score, AUROC
- **Structural bias**: Confidence distribution across item popularity groups; impact on exposure distribution

**Key finding**: Raw LLM confidence is informative but miscalibrated, and exhibits systematic differences across popularity strata — meaning it cannot be used directly as a decision signal without correction.

### Stage 3 — Post-Hoc Calibration

Implemented a post-hoc calibration module (Platt scaling and isotonic regression) with strict valid/test split to prevent information leakage. Calibration visibly improves the alignment between stated confidence and empirical accuracy. Defines `uncertainty = 1 − calibrated_confidence` as the working representation for downstream use.

### Stage 4 — Uncertainty-Aware Reranking

Implemented an uncertainty-penalized ranking module:

$$\text{score}(i) = \hat{c}_i - \lambda \cdot u_i$$

where $\hat{c}_i$ is the calibrated confidence and $u_i = 1 - \hat{c}_i$ is the associated uncertainty. High-uncertainty candidates are systematically downweighted. Evaluation covers both standard ranking metrics (HR@K, NDCG@K, MRR@K) and exposure bias metrics (head/tail item distribution in top-K). Results show uncertainty can meaningfully reshape recommendation lists without collapsing discriminative quality.

### Stage 5 — Multi-Source Uncertainty Modeling

Extended beyond verbalized confidence to a richer uncertainty representation:

- **Verbalized confidence**: Model self-reported score (raw and calibrated)
- **Consistency-based uncertainty**: Estimated via repeated sampling — measuring output agreement, vote distribution, and response stability across multiple inference passes
- **Unified comparison framework**: All estimators evaluated on calibration quality, ranking effectiveness, and behavioral patterns

This stage reframes uncertainty as a multi-dimensional, composable quantity rather than a single scalar.

### Stage 6 — Robustness Evaluation

Built a robustness evaluation framework that runs clean and noisy inputs through the identical inference → calibration → reranking pipeline, comparing degradation across ranking, calibration, and bias metrics. Current experiments use small-scale data where perturbation effects are modest, but the framework is fully operational and ready for larger-scale stress testing.

---

## Current Status

> **Prototype and core experimentation complete. All key modules implemented and validated.**

The full research chain is operational end-to-end. All modules — inference, diagnosis, calibration, reranking, multi-source uncertainty, and robustness — have been implemented and tested. Initial experimental results confirm the core claims: LLM confidence is informative but miscalibrated, calibration improves it, and uncertainty-aware reranking changes system behavior in interpretable ways.

**Current limitation**: Experiments are conducted on small-scale datasets. Findings should be interpreted as proof-of-concept and method validation rather than final empirical conclusions. Some calibration curves show instability due to limited sample counts; robustness differences between baseline and uncertainty-aware methods are not yet fully amplified. Migration to larger, more realistic datasets is the next priority.

---

## Limitations

- Small dataset scale limits statistical stability of calibration and robustness results
- Isotonic regression shows discretization artifacts in low-sample regions
- Robustness experiments do not yet produce large enough performance gaps to draw strong comparative conclusions
- Multi-source uncertainty fusion strategies are currently simple; adaptive or learned fusion is not yet implemented

---

## Future Work

**Scale**: Migrate the full framework to larger-scale recommendation datasets and more realistic evaluation settings to strengthen statistical claims and test generalization.

**Uncertainty modeling**: Explore adaptive fusion strategies across uncertainty estimators; investigate risk-sensitive ranking conditioned on user or item attributes; experiment with dynamic λ scheduling in the reranking objective.

**Robustness**: Introduce stronger input perturbation mechanisms and more complex noise models; conduct sensitivity analyses across noise intensities to characterize the stability boundaries of uncertainty-aware methods.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Preprocessed samples, candidate sets
│   └── popularity_stats/       # Item popularity and exposure statistics
│
├── inference/
│   ├── prompt_builder.py       # Prompt construction from structured samples
│   ├── llm_caller.py           # LLM API interface and multi-sample support
│   └── output_parser.py        # JSON output parsing and prediction logging
│
├── uncertainty/
│   ├── verbalized.py           # Raw confidence extraction and cleaning
│   ├── consistency.py          # Consistency-based uncertainty via repeated sampling
│   ├── logprob_proxy.py        # Log-probability based uncertainty proxy
│   └── calibration.py          # Platt scaling and isotonic regression calibrators
│
├── methods/
│   ├── baseline_ranking.py     # Standard confidence-based ranking
│   └── uncertainty_reranking.py  # Uncertainty-penalized reranking
│
├── evaluation/
│   ├── ranking_metrics.py      # HR@K, NDCG@K, MRR@K
│   ├── calibration_metrics.py  # ECE, Brier score, reliability diagrams
│   ├── bias_metrics.py         # Exposure distribution analysis
│   └── robustness_metrics.py   # Clean vs. noisy degradation analysis
│
├── analysis/
│   └── diagnostics.py          # Confidence–correctness correlation, AUROC, visualization
│
├── experiments/
│   ├── run_inference.py        # Inference pipeline entry point
│   ├── run_calibration.py      # Calibration experiment entry point
│   ├── run_reranking.py        # Reranking experiment entry point
│   ├── run_uncertainty_comparison.py  # Multi-source uncertainty evaluation
│   └── run_robustness.py       # Robustness evaluation entry point
│
├── configs/                    # Experiment configuration files
├── outputs/                    # Prediction files, metrics, figures
└── README.md
```

---

## Key Research Questions

This project is organized around a sequence of connected empirical questions:

1. **Is LLM confidence reliable?** → Diagnosis shows it is informative but systematically miscalibrated
2. **Can miscalibration be corrected?** → Post-hoc calibration improves confidence–accuracy alignment
3. **Can uncertainty improve recommendations?** → Uncertainty-aware reranking reshapes lists in interpretable ways
4. **Is uncertainty multi-dimensional?** → Multiple estimators exhibit distinct calibration profiles and behaviors
5. **Does uncertainty support robust decisions?** → Robustness framework is operational; large-scale validation pending

---

## Citation

If you use this codebase or build on this framework, please cite accordingly. *(Citation information to be added upon publication.)*

---

*This project is currently in the research prototype stage. All core modules are implemented and validated on small-scale data. Large-scale experiments and paper submission are in preparation.*