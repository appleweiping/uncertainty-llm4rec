# Uncertainty-Aware LLMs for Recommendation

> Calibration, Structural Bias, and Decision-Level Uncertainty Integration

A systematic research framework for diagnosing, correcting, and operationalizing uncertainty in LLM-based recommendation — from raw confidence to decision-level integration across a closed-loop evidence chain.

---

## Project Status — April 3, 2025

> **Current Phase: Data-to-Sample Pipeline Complete → Inference Ready**

As of April 3, the project has completed the full data-to-sample pipeline, transforming real-world recommendation data (Amazon Beauty) into a unified pointwise LLM input format. This establishes the necessary foundation for subsequent uncertainty modeling, calibration, and decision-aware recommendation.

The system has successfully established a **unified data-to-LLM interface**, enabling recommendation tasks to be reformulated as structured natural language decision problems.

| Layer | Module | Status |
|---|---|---|
| Data ingestion & normalization | `raw_loaders.py`, `amazon_beauty.yaml` | ✅ Complete |
| Popularity statistics | `popularity_stats.csv` | ✅ Complete |
| Pointwise sample construction | `sample_builder.py` | ✅ Complete |
| Negative sampling | `candidate_sampling.py` | ✅ Complete |
| LLM input schema | Stable `pointwise` format | ✅ Complete |
| LLM inference | `run_inference.py` | ⏳ Pending |
| Confidence extraction & calibration | `run_calibration.py` | ⏳ Pending |
| Uncertainty-aware reranking | `run_reranking.py` | ⏳ Pending |
| Evaluation (ECE, AUROC, NDCG) | `evaluation/` | ⏳ Pending |

---

## Abstract

Large language models (LLMs) are increasingly deployed for recommendation tasks, producing natural language judgments alongside verbalized confidence scores. Yet this confidence signal is rarely subjected to rigorous scrutiny: it goes unmeasured, uncorrected, and unexploited. This work addresses that gap.

We systematically investigate whether LLM-expressed confidence can function as a reliable decision signal in recommendation settings. We demonstrate that it cannot — not in raw form. LLM verbalized confidence is informative but systematically miscalibrated, structurally biased toward popular items, and cannot responsibly guide downstream decisions without correction. At the same time, we establish that this signal is far from worthless: once corrected through post-hoc calibration, it can be transformed into a principled uncertainty estimate and integrated directly into recommendation ranking.

The central contribution is not a new model, but a **complete research framework** — organized as a closed-loop evidence chain spanning inference, diagnosis, calibration, uncertainty-aware reranking, multi-source uncertainty modeling, and robustness evaluation.

---

## Problem Statement

When an LLM assigns "90% confidence" to a recommendation, what does that number actually mean? In most deployments, very little. LLM verbalized confidence exhibits three compounding failure modes:

**Miscalibration.** Stated confidence does not correspond to empirical prediction accuracy. The relationship between expressed certainty and actual correctness is systematic but distorted.

**Structural Bias.** Confidence correlates with item popularity, creating measurable disparities across exposure groups — reflecting structural patterns in pretraining that actively distort recommendation exposure.

**Decision Underutilization.** Even when informative, confidence is rarely integrated into the ranking or selection process — missing a principled opportunity to make it a first-class decision variable.

---

## Research Questions

| # | Research Question | Finding |
|---|---|---|
| **RQ1** | Is LLM verbalized confidence a reliable decision signal? | Informative but systematically miscalibrated; exhibits structural bias across popularity strata |
| **RQ2** | Can miscalibration be corrected through post-hoc methods? | Yes — Platt scaling and isotonic regression meaningfully improve confidence–accuracy alignment |
| **RQ3** | Can corrected uncertainty improve recommendation quality? | Uncertainty-aware reranking reshapes lists interpretably without collapsing ranking metrics |
| **RQ4** | Is uncertainty a multi-dimensional, composable quantity? | Distinct estimators capture different facets of model uncertainty with different behavioral profiles |
| **RQ5** | Does uncertainty-awareness confer robustness to input noise? | Framework validated end-to-end; large-scale effect characterization is ongoing |

---

## Method Pipeline

The project follows a closed-loop evidence chain. Each stage is epistemically dependent on the previous: calibration is validated against diagnosed pathologies; reranking operates on calibrated outputs; robustness is measured over the complete reranking pipeline.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 1  │  Pointwise LLM Inference                    [⏳ Next]    │
│           │  Prompt construction → LLM API → JSON parsing → Logs    │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 2  │  Confidence Diagnosis                        [⏳ Next]    │
│           │  Calibration quality · Discrimination · Bias audit       │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 3  │  Post-Hoc Calibration                        [⏳ Next]    │
│           │  Platt scaling · Isotonic regression · Leakage-free      │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 4  │  Uncertainty-Aware Reranking                 [⏳ Next]    │
│           │  score(i) = ĉᵢ − λ·uᵢ  ·  HR@K, NDCG@K, MRR@K, Bias   │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 5  │  Multi-Source Uncertainty Modeling           [⏳ Next]    │
│           │  Verbalized · Consistency-based · Unified comparison     │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 6  │  Robustness Evaluation                       [⏳ Next]    │
│           │  Clean vs. perturbed · Full pipeline degradation audit   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Completed Work (April 3)

### Stage 0 — Data-to-Sample Pipeline ✅

This is the foundation for all downstream stages. The core contribution here is not data processing per se, but the **reformulation of the recommendation problem as a structured LLM decision task**.

#### Data Ingestion & Normalization

```
configs/data/amazon_beauty.yaml     # Config-driven data loading
src/data/raw_loaders.py             # Unified Amazon review + metadata processing
                                    # Resolves asin / parent_asin schema conflict
main_preprocess.py                  # raw → processed pipeline entry point
```

Outputs: `interactions.csv`, `items.csv`, `users.csv`, `popularity_stats.csv`

#### Pointwise Sample Construction

```
src/data/sample_builder.py          # Leave-one-out split, positive sample expansion
src/data/candidate_sampling.py      # Reproducible negative sampling
main_build_samples.py               # Pipeline entry point → train/valid/test.jsonl
```

#### Stable Pointwise Schema

Every downstream module — prompt builder, inference, calibration, evaluation — consumes this format without modification:

```json
{
  "user_id": "...",
  "history": "...",
  "candidate_item_id": "...",
  "candidate_title": "...",
  "candidate_text": "...",
  "label": 0,
  "target_popularity_group": "head | mid | tail",
  "timestamp": "..."
}
```

This schema encodes the key research design decisions: user history as context, candidate item as a yes/no judgment target, and popularity group as a structural variable for bias analysis.

---

## Experimental Design (Planned)

### Stage 1 — Pointwise Inference Loop

Structured user-item samples → natural language prompts → LLM API → structured prediction records containing `recommend`, `confidence`, `reason`, and raw response. Every subsequent stage traces back to the prediction files produced here.

### Stage 2 — Confidence Diagnosis

Three-dimensional analysis of raw LLM confidence:

- **Discrimination**: AUROC, conditional accuracy at high/low confidence thresholds
- **Calibration Quality**: ECE, Brier score, reliability diagrams
- **Structural Bias Audit**: Confidence distributions decomposed across popularity strata; propagation into exposure bias in top-K lists

### Stage 3 — Post-Hoc Calibration

Platt scaling and isotonic regression under strict valid/test split. Working uncertainty representation:

$$u_i = 1 - \hat{c}_i$$

where $\hat{c}_i$ is calibrated confidence for item $i$.

### Stage 4 — Uncertainty-Aware Reranking

Uncertainty-penalized ranking objective:

$$\text{score}(i) = \hat{c}_i - \lambda \cdot u_i$$

Evaluated across HR@K, NDCG@K, MRR@K, and exposure bias metrics.

### Stage 5 — Multi-Source Uncertainty Modeling

- **Verbalized Confidence**: Model self-reported score, raw and calibrated
- **Consistency-Based Uncertainty**: Vote entropy and response stability across repeated sampling passes

### Stage 6 — Robustness Evaluation

Clean vs. perturbed inputs through the identical inference → calibration → reranking pipeline, tracking degradation simultaneously across ranking, calibration, and exposure bias metrics.

---

## Key Contributions

**C1 — Unified uncertainty-aware recommendation framework.** End-to-end pipeline treating LLM confidence as a first-class variable across its full lifecycle.

**C2 — Systematic empirical characterization of LLM confidence pathologies.** Structured diagnostic analysis quantifying discrimination, calibration quality, and structural bias.

**C3 — Operationalization of uncertainty as a decision variable.** The critical step from measurement to mechanism — uncertainty as an active component of the recommendation decision, not just a diagnostic quantity.

**C4 — Multi-source uncertainty characterization.** Unified evaluation framework demonstrating that distinct estimators exhibit meaningfully different calibration profiles and ranking behaviors.

**C5 — Robustness evaluation pipeline.** Framework for evaluating recommendation systems under input perturbation, tracking degradation across the full pipeline simultaneously.

---

## Preliminary Findings

*(From proof-of-concept experiments — formal results pending at scale)*

- LLM confidence is informative but miscalibrated: above-chance AUROC with systematic ECE gap
- Structural bias is present and measurable: high-confidence scores concentrate on popular items, propagating into exposure concentration in top-K lists
- Post-hoc calibration meaningfully improves confidence–accuracy alignment
- Uncertainty-aware reranking reshapes recommendation lists without catastrophic degradation of ranking metrics
- Distinct uncertainty estimators (verbalized vs. consistency-based) do not converge to the same signal — uncertainty is multi-dimensional

---

## Limitations

- **Dataset scale.** All current experiments on small-scale data; results should be interpreted as method validation, not final empirical conclusions.
- **Calibration artifacts.** Isotonic regression produces discretization artifacts in low-sample confidence regions.
- **Robustness signal strength.** Current perturbation magnitudes insufficient to differentiate methods conclusively at this scale.
- **Uncertainty fusion.** Multi-source combination currently relies on simple aggregation; learned fusion strategies not yet implemented.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Preprocessed samples, candidate sets
│   └── popularity_stats/       # Item popularity and exposure statistics
│
├── configs/
│   └── data/
│       └── amazon_beauty.yaml  # Config-driven data loading
│
├── src/
│   └── data/
│       ├── raw_loaders.py      # Amazon review + metadata ingestion
│       ├── sample_builder.py   # Leave-one-out split, pointwise construction
│       └── candidate_sampling.py  # Reproducible negative sampling
│
├── inference/
│   ├── prompt_builder.py       # Prompt construction from structured samples
│   ├── llm_caller.py           # LLM API interface with multi-sample support
│   └── output_parser.py        # JSON output parsing and structured logging
│
├── uncertainty/
│   ├── verbalized.py           # Raw confidence extraction and normalization
│   ├── consistency.py          # Consistency-based uncertainty via repeated sampling
│   ├── logprob_proxy.py        # Log-probability uncertainty proxy
│   └── calibration.py          # Platt scaling and isotonic regression calibrators
│
├── methods/
│   ├── baseline_ranking.py     # Standard confidence-based ranking baseline
│   └── uncertainty_reranking.py  # Uncertainty-penalized reranking
│
├── evaluation/
│   ├── ranking_metrics.py      # HR@K, NDCG@K, MRR@K
│   ├── calibration_metrics.py  # ECE, Brier score, reliability diagrams
│   ├── bias_metrics.py         # Exposure distribution across popularity strata
│   └── robustness_metrics.py   # Clean vs. perturbed degradation analysis
│
├── analysis/
│   └── diagnostics.py          # Confidence–correctness analysis, AUROC, visualization
│
├── experiments/
│   ├── main_preprocess.py      # Stage 0: data preprocessing
│   ├── main_build_samples.py   # Stage 0: sample construction
│   ├── run_inference.py        # Stage 1 entry point
│   ├── run_calibration.py      # Stage 3 entry point
│   ├── run_reranking.py        # Stage 4 entry point
│   ├── run_uncertainty_comparison.py  # Stage 5 entry point
│   └── run_robustness.py       # Stage 6 entry point
│
└── outputs/                    # Prediction files, metrics, figures
```

---

## Future Work

**Scale and generalization.** Migrating the full framework to larger, more realistic recommendation datasets. Statistical stability across all stages is contingent on sufficient sample counts.

**Uncertainty modeling.** Adaptive and learned fusion strategies across uncertainty estimators; risk-sensitive ranking objectives; dynamic λ scheduling conditioned on local uncertainty distribution.

**Robustness.** Stronger and more structured input perturbation mechanisms: feature-level corruption, interaction history noise, adversarial prompt perturbations. Systematic sensitivity analyses across noise intensities.

**Uncertainty as a training signal.** Investigating whether uncertainty estimates can inform model training itself — uncertainty-weighted loss, calibration-aware fine-tuning, or active learning over uncertain user-item pairs.

---

*All core data and sample pipeline modules are implemented and validated. LLM inference, calibration, reranking, and evaluation are in active development. Current results constitute method validation and pipeline proof-of-concept; empirical conclusions at scale are forthcoming.*