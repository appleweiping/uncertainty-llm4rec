# Uncertainty-Aware LLMs for Recommendation

> Calibration, Structural Bias, and Decision-Level Uncertainty Integration

A research framework for diagnosing, correcting, and operationalizing uncertainty in LLM-based recommendation — from raw confidence scores to uncertainty-aware ranking across a closed-loop evidence chain.

---

## Overview

When an LLM assigns "90% confidence" to a recommendation, what does that number actually mean? This project systematically investigates whether LLM-expressed confidence can function as a reliable decision signal in recommendation settings. The short answer: **not in raw form**.

LLM verbalized confidence is informative but systematically miscalibrated, structurally biased toward popular items, and cannot responsibly guide downstream decisions without correction. This work builds a complete pipeline to diagnose these pathologies, apply post-hoc calibration, and integrate corrected uncertainty directly into recommendation ranking.

**Central question:** Can LLM confidence be treated as a reliable signal in recommendation systems?

---

## Research Questions

| # | Question | Status |
|---|----------|--------|
| RQ1 | Is LLM verbalized confidence a reliable decision signal? | Informative but miscalibrated; exhibits popularity bias |
| RQ2 | Can miscalibration be corrected through post-hoc methods? | Platt scaling and isotonic regression improve alignment |
| RQ3 | Can corrected uncertainty improve recommendation quality? | Uncertainty-aware reranking reshapes lists without collapsing metrics |
| RQ4 | Is uncertainty a multi-dimensional, composable quantity? | Distinct estimators capture different facets of uncertainty |
| RQ5 | Does uncertainty-awareness confer robustness to input noise? | Pipeline validated; large-scale characterization ongoing |

---

## Method Pipeline

Each stage is epistemically dependent on the previous: calibration is validated against diagnosed pathologies; reranking operates on calibrated outputs; robustness is measured over the complete pipeline.

```
Stage 0  │  Data-to-Sample Pipeline                    [✅ Complete]
         │  Raw Amazon data → preprocessing → pointwise LLM input format

Stage 1  │  Pointwise LLM Inference                    [⏳ Pending]
         │  Prompt construction → LLM API → JSON parsing → prediction logs

Stage 2  │  Confidence Diagnosis                        [⏳ Pending]
         │  Calibration quality · Discrimination · Popularity bias audit

Stage 3  │  Post-Hoc Calibration                        [⏳ Pending]
         │  Platt scaling · Isotonic regression · Leakage-free split

Stage 4  │  Uncertainty-Aware Reranking                 [⏳ Pending]
         │  score(i) = ĉᵢ − λ·uᵢ  ·  HR@K, NDCG@K, MRR@K, Bias

Stage 5  │  Multi-Source Uncertainty Modeling           [⏳ Pending]
         │  Verbalized · Consistency-based · Unified comparison

Stage 6  │  Robustness Evaluation                       [⏳ Pending]
         │  Clean vs. perturbed · Full pipeline degradation audit
```

---

## Key Design Decisions

**Correct task formulation.** Candidate items must not appear in the user's interaction history. This prevents data leakage, trivial predictions, and artificially inflated confidence scores — reformulating the task as true *new item recommendation*.

**Unified pointwise schema.** Every downstream module consumes a single stable format:

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

**Uncertainty as a decision variable.** Working uncertainty representation: `uᵢ = 1 − ĉᵢ`, where `ĉᵢ` is calibrated confidence. Uncertainty-penalized ranking: `score(i) = ĉᵢ − λ·uᵢ`.

---

## Preliminary Findings

*(From proof-of-concept experiments on Amazon Beauty — formal results pending)*

- LLM confidence is **informative but miscalibrated**: above-chance AUROC with systematic ECE gap
- **Structural bias is measurable**: high-confidence scores concentrate on popular items, propagating into exposure concentration in top-K lists
- High-confidence predictions (0.8–0.9) often have low empirical accuracy (~0.15–0.2); reliability curve is far below the diagonal
- Post-hoc calibration meaningfully improves confidence–accuracy alignment
- Uncertainty-aware reranking reshapes recommendation lists without catastrophic degradation of ranking metrics
- Verbalized and consistency-based uncertainty estimators do not converge to the same signal — uncertainty is multi-dimensional

---

## Repository Structure

```
.
├── configs/
│   └── data/
│       └── amazon_beauty.yaml          # Config-driven data loading
│
├── src/
│   └── data/
│       ├── raw_loaders.py              # Amazon review + metadata ingestion
│       ├── sample_builder.py           # Leave-one-out split, pointwise construction
│       └── candidate_sampling.py       # Reproducible negative sampling
│
├── inference/
│   ├── prompt_builder.py               # Prompt construction from structured samples
│   ├── llm_caller.py                   # LLM API interface
│   └── output_parser.py                # JSON output parsing and structured logging
│
├── uncertainty/
│   ├── verbalized.py                   # Raw confidence extraction and normalization
│   ├── consistency.py                  # Consistency-based uncertainty
│   ├── logprob_proxy.py                # Log-probability uncertainty proxy
│   └── calibration.py                  # Platt scaling and isotonic regression
│
├── methods/
│   ├── baseline_ranking.py             # Standard confidence-based ranking
│   └── uncertainty_reranking.py        # Uncertainty-penalized reranking
│
├── evaluation/
│   ├── ranking_metrics.py              # HR@K, NDCG@K, MRR@K
│   ├── calibration_metrics.py          # ECE, Brier score, reliability diagrams
│   ├── bias_metrics.py                 # Exposure distribution across popularity strata
│   └── robustness_metrics.py           # Clean vs. perturbed degradation analysis
│
├── analysis/
│   └── diagnostics.py                  # Confidence–correctness analysis, AUROC, visualization
│
├── experiments/
│   ├── main_preprocess.py              # Stage 0: data preprocessing
│   ├── main_build_samples.py           # Stage 0: sample construction
│   ├── run_inference.py                # Stage 1
│   ├── run_calibration.py              # Stage 3
│   ├── run_reranking.py                # Stage 4
│   ├── run_uncertainty_comparison.py   # Stage 5
│   └── run_robustness.py               # Stage 6
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── popularity_stats/
│
└── outputs/                            # Prediction files, metrics, figures
```

---

## Running the Pipeline

**Stage 0a — Data preprocessing**
```bash
python main_preprocess.py --config configs/data/amazon_beauty.yaml
```
Outputs: `interactions.csv`, `items.csv`, `users.csv`, `popularity_stats.csv`

**Stage 0b — Sample construction**
```bash
python main_build_samples.py --config configs/data/amazon_beauty.yaml
```
Outputs: `train.jsonl`, `valid.jsonl`, `test.jsonl`

**Stage 1 — LLM inference**
```bash
python run_inference.py --config configs/exp/beauty_deepseek.yaml
```
LLM output format:
```json
{
  "recommend": "yes/no",
  "confidence": 0.85,
  "reason": "..."
}
```

**Stage 2+ — Evaluation and calibration**
```bash
python run_calibration.py
python run_reranking.py
python run_uncertainty_comparison.py
python run_robustness.py
```

---

## Limitations

- **Dataset scale.** All current experiments on small-scale data; results should be interpreted as method validation, not final empirical conclusions.
- **Calibration artifacts.** Isotonic regression produces discretization artifacts in low-sample confidence regions.
- **Robustness signal strength.** Current perturbation magnitudes insufficient to differentiate methods conclusively at this scale.
- **Uncertainty fusion.** Multi-source combination currently relies on simple aggregation; learned fusion not yet implemented.

---

## Status

*Data and sample pipeline: complete. LLM inference, calibration, reranking, and evaluation: in active development. Current results constitute method validation and pipeline proof-of-concept; empirical conclusions at scale are forthcoming.*