# Uncertainty-Aware LLM Recommendation

> Diagnosing, calibrating, and operationalizing confidence in LLM-based recommendation.

This repository studies a simple but important question: when a large language model says it is confident about a recommendation, can that confidence be trusted as a decision signal?

Our answer is structured as a full pipeline rather than a single metric. We first diagnose whether verbalized confidence is informative, calibrated, and behaviorally biased. We then convert raw confidence into calibrated confidence, derive uncertainty from it, and finally test whether uncertainty can be used at decision time through lightweight reranking. The goal is not only to analyze confidence, but to make it usable.

## Overview

The project is organized as an evidence chain:

1. Build a clean pointwise recommendation task from sequential interaction data.
2. Query an LLM for recommendation decisions with verbalized confidence.
3. Diagnose confidence quality through correctness, calibration, and popularity-related bias.
4. Fit a post-hoc calibrator on validation predictions and apply it to test predictions.
5. Use calibrated uncertainty in reranking and evaluate both utility and exposure behavior.
6. Extend the framework toward richer uncertainty estimators and robustness analysis.

This progression matters. Calibration is only meaningful after diagnosis, and reranking is only meaningful once uncertainty is defined in a leakage-free way.

## Research Scope

The repository is designed around four connected research questions:

- Is LLM confidence informative in recommendation, or merely stylistic?
- How miscalibrated is verbalized confidence, and can it be corrected post hoc?
- Can calibrated uncertainty influence downstream ranking behavior in a controlled way?
- Does uncertainty-awareness change not only ranking metrics, but also exposure patterns across head and long-tail items?

The current implementation focuses on method validation and pipeline integrity. It is meant to support clean empirical iteration rather than overclaiming final conclusions from small-scale runs.

## Method Pipeline

At a high level, the implemented workflow is:

```text
Preprocess -> Build Samples -> Inference -> Evaluation -> Calibration -> Reranking
```

The current method layer includes:

- Confidence extraction from LLM outputs
- Calibration via standard post-hoc methods such as isotonic regression and Platt scaling
- Leakage-aware calibration protocol: fit on `valid`, apply on `test`
- Uncertainty-aware reranking with a minimal decision rule built on calibrated confidence
- Evaluation over both ranking quality and distributional behavior

This repository intentionally favors standard, interpretable baselines before more complex uncertainty modeling.

## What Is Implemented

The codebase already supports the core week-one research loop:

- Data preprocessing and pointwise sample construction
- LLM inference with configurable backends
- Confidence parsing and normalization
- Calibration diagnostics, including ECE and Brier score
- Strict calibration with separate validation and test prediction files
- Reranking evaluation with ranking metrics and bias-oriented metrics
- Initial scaffolding for richer uncertainty estimators, including consistency-based and fused variants

In other words, the project has moved beyond pure diagnosis and into the first decision-level uncertainty pipeline.

## Repository Layout

```text
.
|-- configs/                  # data, model, and experiment configurations
|-- data/                     # raw and processed datasets
|-- outputs/                  # predictions, calibrated outputs, tables, and figures
|-- prompts/                  # LLM prompting templates
|-- scripts/                  # convenience scripts for staged runs
|-- src/
|   |-- analysis/             # diagnostic analysis and plotting
|   |-- data/                 # preprocessing, sample construction, noise, popularity
|   |-- eval/                 # ranking, calibration, bias, and robustness metrics
|   |-- llm/                  # backends, prompting, parsing, inference
|   |-- methods/              # baseline ranking and uncertainty-aware reranking
|   |-- uncertainty/          # confidence extraction, calibration, estimator variants
|   `-- utils/                # IO, logging, paths, registry, seeding
|-- main_preprocess.py
|-- main_build_samples.py
|-- main_infer.py
|-- main_eval.py
|-- main_calibrate.py
|-- main_rerank.py
`-- main_uncertainty_compare.py
```

## Environment

Use the project with Python 3.12. In this repository, the safest convention is to avoid ambiguous `python` calls and instead use `py -3.12` or the project virtual environment explicitly.

Minimal setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you are using the DeepSeek backend, set the API key in the environment before inference:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
```

## Quickstart

The commands below show the intended end-to-end flow on Amazon Beauty.

### 1. Preprocess raw data

```powershell
py -3.12 main_preprocess.py --config configs/data/amazon_beauty.yaml
```

### 2. Build pointwise train/valid/test samples

```powershell
py -3.12 main_build_samples.py --config configs/data/amazon_beauty.yaml
```

### 3. Run LLM inference

Generate split-specific prediction files:

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek.yaml `
  --input_path data/processed/amazon_beauty/valid.jsonl `
  --output_path outputs/beauty_deepseek/predictions/valid_raw.jsonl `
  --split_name valid `
  --overwrite
```

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek.yaml `
  --input_path data/processed/amazon_beauty/test.jsonl `
  --output_path outputs/beauty_deepseek/predictions/test_raw.jsonl `
  --split_name test `
  --overwrite
```

### 4. Evaluate prediction quality

```powershell
py -3.12 main_eval.py --exp_name beauty_deepseek
```

### 5. Run strict calibration

Calibration is designed to be leakage-aware:

- fit on `valid_raw.jsonl`
- apply on `test_raw.jsonl`
- output calibrated confidence and uncertainty

```powershell
py -3.12 main_calibrate.py `
  --exp_name beauty_deepseek `
  --valid_path outputs/beauty_deepseek/predictions/valid_raw.jsonl `
  --test_path outputs/beauty_deepseek/predictions/test_raw.jsonl `
  --method isotonic
```

### 6. Run uncertainty-aware reranking

```powershell
py -3.12 main_rerank.py `
  --exp_name beauty_deepseek `
  --input_path outputs/beauty_deepseek/calibrated/test_calibrated.jsonl `
  --lambda_penalty 0.5
```

## Key Outputs

Typical experiment artifacts are written under `outputs/{exp_name}/`:

- `predictions/valid_raw.jsonl`
- `predictions/test_raw.jsonl`
- `calibrated/test_calibrated.jsonl`
- `tables/calibration_comparison.csv`
- `tables/calibration_split_metadata.csv`
- `tables/rerank_results.csv`

These files are enough to audit whether calibration is leakage-free, whether calibration improves reliability, and whether uncertainty-aware reranking changes ranking or exposure behavior.

## Evaluation Philosophy

The project does not treat ranking quality as the only outcome. We evaluate two classes of effects together:

- Ranking utility: HR@K, NDCG@K, MRR
- Distributional behavior: head exposure ratio and long-tail coverage

This is deliberate. A method that appears stable in ranking metrics may still change who receives exposure, and a method that slightly reshapes ranking may still be valuable if it produces more reliable decision behavior.

## Current Status

The repository already supports:

- clean data-to-sample construction
- end-to-end inference and evaluation
- strict validation-to-test calibration
- first-pass uncertainty-aware reranking

Current experiments are best understood as method-grounding and pipeline validation. The next natural extensions are broader domain transfer, richer uncertainty estimators, and stronger robustness experiments.

## Notes

- Current small-scale runs are useful for validating methodology, not for claiming final large-scale empirical conclusions.
- The first reranking variant is intentionally conservative and interpretable.
- The codebase is structured so that new uncertainty estimators can be added without rewriting the evaluation chain.

## License

This project is released under the terms of the [LICENSE](LICENSE).
