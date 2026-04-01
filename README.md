# Uncertainty-Aware LLMs for Recommendation
### Calibration, Structural Bias, and Decision-Level Uncertainty Integration

> A systematic research framework for diagnosing, correcting, and operationalizing uncertainty in LLM-based recommendation — from raw confidence to decision-level integration across a closed-loop evidence chain.

---

## Abstract

Large language models (LLMs) are increasingly deployed for recommendation tasks, producing natural language judgments alongside verbalized confidence scores. Yet this confidence signal is rarely subjected to rigorous scrutiny: it goes unmeasured, uncorrected, and unexploited. This work addresses that gap.

We systematically investigate whether LLM-expressed confidence can function as a reliable decision signal in recommendation settings. We demonstrate that it cannot — not in raw form. LLM verbalized confidence is informative but systematically miscalibrated, structurally biased toward popular items, and cannot responsibly guide downstream decisions without correction. At the same time, we establish that this signal is far from worthless: once corrected through post-hoc calibration, it can be transformed into a principled uncertainty estimate and integrated directly into recommendation ranking, reshaping system behavior in measurable, interpretable ways.

The central contribution of this work is not a new model, but a **complete research framework** — organized as a closed-loop evidence chain spanning inference, diagnosis, calibration, uncertainty-aware reranking, multi-source uncertainty modeling, and robustness evaluation. Each stage produces structured outputs that condition the next, transforming uncertainty from an analytical observation into an active decision variable.

---

## Problem Statement

When an LLM assigns "90% confidence" to a recommendation, what does that number actually mean? In most deployments, very little. LLM verbalized confidence exhibits three compounding failure modes that render it unsuitable as a direct decision signal:

**Miscalibration.** Stated confidence does not correspond to empirical prediction accuracy. The relationship between expressed certainty and actual correctness is systematic but distorted — requiring explicit correction before the signal can be trusted.

**Structural Bias.** Confidence is not uniformly distributed across the item space. It correlates with item popularity, creating measurable disparities across exposure groups. This is not a random artifact: it reflects structural patterns in pretraining and prompting that actively distort recommendation exposure, advantaging already-popular items in ways that compound existing feedback loops.

**Decision Underutilization.** Even when informative, confidence is rarely integrated into the ranking or selection process. Uncertainty exists as a measurement but not as a mechanism — missing a principled opportunity to make it a first-class decision variable.

These failure modes motivate a systematic research agenda: define uncertainty rigorously, measure its pathologies, correct for them, and integrate the result into the recommendation decision itself.

---

## Research Questions

This project is organized around a progressive sequence of connected empirical questions. Each question is answered by a dedicated experimental stage; each answer conditions the formulation of the next. The chain is cumulative, not additive.

| # | Research Question | Finding |
|---|---|---|
| **RQ1** | Is LLM verbalized confidence a reliable decision signal? | Informative but systematically miscalibrated; exhibits structural bias across popularity strata |
| **RQ2** | Can miscalibration be corrected through post-hoc methods? | Yes — Platt scaling and isotonic regression meaningfully improve confidence–accuracy alignment |
| **RQ3** | Can corrected uncertainty improve recommendation quality? | Uncertainty-aware reranking reshapes lists interpretably without collapsing ranking metrics |
| **RQ4** | Is uncertainty a multi-dimensional, composable quantity? | Distinct estimators capture different facets of model uncertainty with different behavioral profiles |
| **RQ5** | Does uncertainty-awareness confer robustness to input noise? | Framework validated end-to-end; large-scale effect characterization is ongoing |

---

## Key Contributions

**C1 — A unified uncertainty-aware recommendation framework.**
We establish a complete, end-to-end research pipeline that treats LLM confidence as a first-class variable across its full lifecycle: from raw inference through diagnosis, calibration, decision integration, multi-source modeling, and robustness evaluation. The framework is modular, reproducible, and designed as a reusable substrate for future research on uncertainty in LLM-based systems.

**C2 — Systematic empirical characterization of LLM confidence pathologies.**
We conduct a structured diagnostic analysis of raw LLM confidence, quantifying its discrimination ability (AUROC), calibration quality (ECE, Brier score, reliability diagrams), and structural bias across item popularity strata. We document, with both metrics and visualizations, that LLM confidence is a corrupted but recoverable signal — worth fixing, not discarding.

**C3 — Operationalization of uncertainty as a decision variable.**
We go beyond treating uncertainty as an analytical observation. We define a calibrated uncertainty estimate (`u = 1 − calibrated confidence`) and integrate it directly into recommendation ranking via an uncertainty-penalized scoring objective. This is the critical step from measurement to mechanism — transforming uncertainty from a diagnostic quantity into an active component of the recommendation decision.

**C4 — Multi-source uncertainty characterization.**
We extend beyond verbalized confidence to a richer uncertainty representation incorporating consistency-based estimates derived from repeated sampling. We establish a unified evaluation framework and demonstrate that distinct estimators exhibit meaningfully different calibration profiles, ranking behaviors, and uncertainty patterns — establishing uncertainty as a multi-dimensional, composable quantity rather than a fixed scalar.

**C5 — A robustness evaluation pipeline for uncertainty-aware recommendation.**
We construct a framework for evaluating recommendation systems under input perturbation, simultaneously tracking degradation across ranking, calibration, and bias metrics through the full pipeline. This establishes an operational foundation for characterizing the stability boundaries of uncertainty-aware methods under realistic noise conditions.

---

## Method Pipeline

The project follows a closed-loop evidence chain. Each stage is epistemically dependent on the previous: calibration is validated against diagnosed pathologies; reranking operates on calibrated outputs; robustness is measured over the complete reranking pipeline. The design ensures that findings at each stage are grounded in, and traceable to, the outputs of prior stages.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 1  │  Pointwise LLM Inference                                 │
│           │  Prompt construction → LLM API → JSON parsing → Logs    │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 2  │  Confidence Diagnosis                                     │
│           │  Calibration quality · Discrimination · Bias audit       │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 3  │  Post-Hoc Calibration                                     │
│           │  Platt scaling · Isotonic regression · Leakage-free      │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 4  │  Uncertainty-Aware Reranking                              │
│           │  score(i) = ĉᵢ − λ·uᵢ  ·  HR@K, NDCG@K, MRR@K, Bias   │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 5  │  Multi-Source Uncertainty Modeling                        │
│           │  Verbalized · Consistency-based · Unified comparison     │
└─────────────────────────────┬────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 6  │  Robustness Evaluation                                    │
│           │  Clean vs. perturbed · Full pipeline degradation audit   │
└──────────────────────────────────────────────────────────────────────┘
```

This is not a collection of independent modules. The pipeline is a coherent evidence chain: each stage both consumes the outputs of and provides grounding for the stages around it.

---

## Experimental Design

### Stage 1 — Pointwise Inference Loop

We construct a minimal, fully reproducible LLM recommendation pipeline. Structured user-item samples are transformed into natural language prompts via a prompt builder, submitted to the LLM through a unified API interface, and parsed into structured prediction records containing `recommend`, `confidence`, `reason`, and the raw model response. This stage establishes the experimental substrate — a completely logged, deterministic inference loop — on which all downstream analysis operates. Every subsequent stage traces back to the prediction files produced here.

### Stage 2 — Confidence Diagnosis

We systematically investigate raw LLM confidence along three dimensions:

- **Discrimination Analysis**: We quantify the relationship between expressed confidence and prediction correctness, characterizing model behavior at high and low confidence thresholds via AUROC and conditional accuracy.
- **Calibration Quality**: We compute Expected Calibration Error (ECE), Brier score, and construct reliability diagrams that render miscalibration visually and numerically explicit.
- **Structural Bias Audit**: We decompose confidence distributions across item popularity strata and measure how confidence disparities propagate into exposure bias in top-K recommendation lists.

This stage establishes the empirical case for correction: the signal is informative but corrupted, and the corruption is structured, not random.

### Stage 3 — Post-Hoc Calibration

We apply post-hoc calibration using two complementary methods — Platt scaling and isotonic regression — under a strict valid/test split to prevent information leakage. Calibrated confidence is validated through pre- and post-calibration reliability diagrams. We define the working uncertainty representation as:

$$u_i = 1 - \hat{c}_i$$

where $\hat{c}_i$ denotes calibrated confidence for item $i$. This operationalization provides the bridge from analytical measurement to decision integration.

### Stage 4 — Uncertainty-Aware Reranking

We implement an uncertainty-penalized ranking objective:

$$\text{score}(i) = \hat{c}_i - \lambda \cdot u_i$$

where $\lambda$ controls the strength of the uncertainty penalty. High-uncertainty candidates are systematically downweighted even when they carry high raw confidence. We evaluate across standard ranking metrics (HR@K, NDCG@K, MRR@K) and exposure bias metrics (head/tail item distribution in top-K). This stage demonstrates that uncertainty, once operationalized, functions as a meaningful decision variable — not merely a diagnostic one.

### Stage 5 — Multi-Source Uncertainty Modeling

We extend the uncertainty representation beyond a single scalar to a multi-source, composable structure:

- **Verbalized Confidence**: Model self-reported score, raw and calibrated.
- **Consistency-Based Uncertainty**: Estimated via repeated sampling — quantifying output agreement, vote entropy, and response stability across multiple inference passes.

We evaluate all estimators within a unified comparison framework, measuring calibration quality, ranking effectiveness, and behavioral patterns. This stage establishes that different estimators capture meaningfully distinct aspects of model uncertainty and cannot be treated as interchangeable.

### Stage 6 — Robustness Evaluation

We construct a robustness evaluation framework that routes both clean and perturbed inputs through the identical inference → calibration → reranking pipeline, comparing performance degradation simultaneously across ranking, calibration, and exposure bias metrics. Perturbations are applied at the input level, simulating realistic noise in user-item interaction signals. The framework is fully operational and architected for large-scale stress testing as dataset scale increases.

---

## Results & Findings

The following summarizes key empirical findings from current proof-of-concept experiments:

- **LLM confidence is informative but miscalibrated.** Reliability diagrams and ECE measurements confirm a systematic gap between expressed and empirical confidence. Above-chance AUROC demonstrates that the signal carries genuine discriminative information — it is not noise, but it is distorted and requires correction.

- **Structural bias is present and measurable.** Confidence distributions differ significantly across item popularity strata. High-confidence scores concentrate on popular items, and this disparity propagates directly into exposure concentration in top-K recommendation lists — a mechanism through which miscalibrated confidence amplifies existing popularity bias.

- **Post-hoc calibration meaningfully improves confidence–accuracy alignment.** Both Platt scaling and isotonic regression reduce ECE and improve reliability diagram alignment. Isotonic regression exhibits discretization artifacts in low-sample confidence regions — a known limitation at small data scales that does not undermine the method's validity at scale.

- **Uncertainty-aware reranking reshapes recommendation lists in interpretable ways.** Applying the uncertainty penalty redistributes item rankings, reducing exposure concentration toward high-confidence (typically popular) items. Standard ranking metrics are not catastrophically degraded, demonstrating that uncertainty integration does not simply trade accuracy for diversity.

- **Distinct uncertainty estimators exhibit different behavioral profiles.** Verbalized confidence and consistency-based uncertainty do not converge to the same signal under evaluation. They exhibit different calibration quality, different ranking behavior, and different sensitivity patterns — confirming that uncertainty is a multi-dimensional quantity and that estimator choice is a substantive research decision.

- **The robustness pipeline is validated end-to-end; large-scale effect characterization is ongoing.** Current small-scale experiments show modest perturbation effects, consistent with low statistical power at this data scale. The framework correctly propagates perturbations through the full pipeline and produces interpretable metric comparisons. Larger datasets are required to characterize the stability boundaries of uncertainty-aware methods under realistic noise conditions.

---

## Limitations

- **Dataset scale.** All current experiments are conducted on small-scale data. Calibration estimates exhibit instability due to limited sample counts, and robustness comparisons do not yet produce effect sizes sufficient for strong comparative conclusions. All reported results should be interpreted as method validation and proof-of-concept, not as final empirical conclusions.
- **Calibration artifacts.** Isotonic regression produces step-function discretization artifacts in low-sample confidence regions, introducing noise into downstream uncertainty estimates at the tails of the confidence distribution.
- **Robustness signal strength.** Current perturbation magnitudes do not produce performance gaps sufficient to differentiate baseline and uncertainty-aware methods conclusively. Stronger noise models and systematic sensitivity analyses across noise intensities are required before stability claims can be made.
- **Uncertainty fusion.** Multi-source uncertainty combination currently relies on simple aggregation. Adaptive or learned fusion strategies — conditioned on item attributes, user context, or estimator reliability — are not yet implemented.

---

## Future Work

**Scale and generalization.** The immediate research priority is migrating the full framework to larger, more realistic recommendation datasets. Statistical stability across all stages — calibration, reranking, robustness — is contingent on sufficient sample counts. Scale is not an enhancement; it is a prerequisite for converting proof-of-concept findings into publishable empirical claims.

**Uncertainty modeling.** We will investigate adaptive and learned fusion strategies across uncertainty estimators, moving beyond simple aggregation toward context-conditioned combination. This includes risk-sensitive ranking objectives conditioned on user or item attributes, and dynamic $\lambda$ scheduling in the reranking objective that adapts penalty strength based on the local uncertainty distribution.

**Robustness.** Future experiments will introduce stronger and more structured input perturbation mechanisms, including feature-level corruption, interaction history noise, and adversarial prompt perturbations. Systematic sensitivity analyses across noise intensities will characterize the stability boundaries of uncertainty-aware methods and identify the conditions under which uncertainty integration provides robustness benefits versus degrades performance.

**Uncertainty as a training signal.** A longer-term direction is investigating whether uncertainty estimates can inform not just post-hoc reranking but model training itself — through uncertainty-weighted loss, calibration-aware fine-tuning, or active learning over uncertain user-item pairs.

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
│   ├── llm_caller.py           # LLM API interface with multi-sample support
│   └── output_parser.py        # JSON output parsing and structured prediction logging
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
│   ├── run_inference.py        # Stage 1 entry point
│   ├── run_calibration.py      # Stage 3 entry point
│   ├── run_reranking.py        # Stage 4 entry point
│   ├── run_uncertainty_comparison.py  # Stage 5 entry point
│   └── run_robustness.py       # Stage 6 entry point
│
├── configs/                    # Experiment configuration files
├── outputs/                    # Prediction files, metrics, figures
└── README.md
```

---

## Citation

If you use this framework or build on this research, please cite accordingly. *(Full citation information will be provided upon publication.)*

---

*All core modules are implemented and validated on small-scale data. Large-scale experiments and paper submission are in preparation. Current results constitute method validation; empirical conclusions at scale are forthcoming.*