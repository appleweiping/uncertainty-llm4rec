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
- Cross-domain validation across Beauty, Movies, Books, and Electronics under the same experimental definition

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

## Paper-Facing Summary Layer

For paper writing, the repository now maintains two result layers:

- experiment-complete summaries under `outputs/summary/`
- Beauty-centered paper-facing tables derived from those summaries

After running:

```powershell
py -3.12 main_aggregate_all.py --output_root outputs
```

the main Beauty-facing exports are:

- `outputs/summary/beauty_main_results.csv`
- `outputs/summary/beauty_estimator_brief.csv`
- `outputs/summary/beauty_robustness_curve_brief.csv`
- `outputs/summary/beauty_reproducibility_brief.csv`

These are intended to be the direct bridge from experiment artifacts to paper tables.

For the current Beauty-first writing phase, the main coordination docs are:

- `docs/paper_outline.md`
- `docs/beauty_freeze_checklist.md`
- `docs/tables.md`

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

## Config Structure

The repository is organized around three config layers:

- `configs/data/`: domain-specific preprocessing and sample-building settings
- `configs/model/`: backend, model, connection, and generation settings
- `configs/exp/`: experiment-level inference settings such as `exp_name`, input path, output root, prompt path, and model config

This design keeps experiments reproducible and makes Week2 extensions easier. In practice:

- changing domains should mostly mean switching `configs/data/*.yaml`
- changing models should mostly mean switching `configs/model/*.yaml`
- changing an experiment run should mostly mean switching `configs/exp/*.yaml`

Current model config files include:

- `configs/model/deepseek.yaml`: active backend used in Week1 experiments
- `configs/model/qwen.yaml`: Qwen API-compatible backend config
- `configs/model/glm.yaml`: GLM API-compatible backend config
- `configs/model/doubao.yaml`: Doubao API-compatible backend config
- `configs/model/kimi.yaml`: Kimi API-compatible backend config
- `configs/model/qwen_local_7b.yaml`: local Hugging Face backend skeleton for Qwen 7B
- `configs/model/llama_local_8b.yaml`: local Hugging Face backend skeleton for Llama 8B

Environment variables are read from model configs via `api_key_env`. Typical examples are:

- `DEEPSEEK_API_KEY`
- `QWEN_API_KEY`
- `DOUBAO_API_KEY`
- `KIMI_API_KEY`

For the new `local_hf` backend skeleton, install `transformers` and `torch` in the runtime environment and point `model_path` / `tokenizer_path` to an available local or Hugging Face checkpoint before running inference.

## Quickstart

The commands below show the intended end-to-end flow on Amazon Beauty and the lightweight cross-domain subsets used for Week1 validation.

For Week2 multi-model validation, keep the same data and pipeline, and only switch `model_config` / `exp_name`. Typical Beauty experiment names are:

- `beauty_deepseek`
- `beauty_qwen`
- `beauty_glm`
- `beauty_doubao`
- `beauty_kimi`

The same pattern now extends to the lightweight cross-domain subsets:

- `movies_small_deepseek`, `movies_small_qwen`, `movies_small_kimi`, `movies_small_doubao`
- `books_small_deepseek`, `books_small_qwen`, `books_small_kimi`, `books_small_doubao`
- `electronics_small_deepseek`, `electronics_small_qwen`, `electronics_small_kimi`, `electronics_small_doubao`

### Beauty End-to-End

#### 1. Preprocess raw data

```powershell
py -3.12 main_preprocess.py --config configs/data/amazon_beauty.yaml
```

#### 2. Build pointwise train/valid/test samples

```powershell
py -3.12 main_build_samples.py --config configs/data/amazon_beauty.yaml
```

#### 3. Run LLM inference

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

#### 4. Evaluate prediction quality

```powershell
py -3.12 main_eval.py --exp_name beauty_deepseek
```

#### 5. Run strict calibration

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

#### 6. Run uncertainty-aware reranking

```powershell
py -3.12 main_rerank.py `
  --exp_name beauty_deepseek `
  --input_path outputs/beauty_deepseek/calibrated/test_calibrated.jsonl `
  --lambda_penalty 0.5
```

### Movies-Small Validation

Week1 Day5 uses a lightweight Movies subset to validate that the same pipeline is not Beauty-specific.

#### 1. Build the Movies-small split data

The full Movies preprocess has already been validated. After preprocess, create a lightweight subset under `data/processed/amazon_movies_small/`, then run the original sample-building pipeline:

```powershell
py -3.12 main_build_samples.py --config configs/data/amazon_movies_small.yaml
```

#### 2. Run inference on valid and test

```powershell
py -3.12 main_infer.py `
  --config configs/exp/movies_small_deepseek.yaml `
  --input_path data/processed/amazon_movies_small/valid.jsonl `
  --output_path outputs/movies_small_deepseek/predictions/valid_raw.jsonl `
  --split_name valid `
  --max_samples 100 `
  --overwrite
```

```powershell
py -3.12 main_infer.py `
  --config configs/exp/movies_small_deepseek.yaml `
  --input_path data/processed/amazon_movies_small/test.jsonl `
  --output_path outputs/movies_small_deepseek/predictions/test_raw.jsonl `
  --split_name test `
  --max_samples 100 `
  --overwrite
```

#### 3. Run evaluation, calibration, and reranking

```powershell
py -3.12 main_eval.py --exp_name movies_small_deepseek --input_path outputs/movies_small_deepseek/predictions/test_raw.jsonl
```

```powershell
py -3.12 main_calibrate.py `
  --exp_name movies_small_deepseek `
  --valid_path outputs/movies_small_deepseek/predictions/valid_raw.jsonl `
  --test_path outputs/movies_small_deepseek/predictions/test_raw.jsonl `
  --method isotonic
```

```powershell
py -3.12 main_rerank.py `
  --exp_name movies_small_deepseek `
  --input_path outputs/movies_small_deepseek/calibrated/test_calibrated.jsonl `
  --lambda_penalty 0.5
```

### Books-Small And Electronics-Small Validation

Books and Electronics follow the same principle used for Movies:

- run full-domain `preprocess` first
- construct a lightweight processed-level subset
- reuse the original downstream pipeline unchanged

#### 1. Run full preprocess

```powershell
py -3.12 main_preprocess.py --config configs/data/amazon_books.yaml
```

```powershell
py -3.12 main_preprocess.py --config configs/data/amazon_electronics.yaml
```

#### 2. Build the small-subset samples

Once `data/processed/amazon_books_small/` and `data/processed/amazon_electronics_small/` have been constructed from the processed full-domain outputs, run:

```powershell
py -3.12 main_build_samples.py --config configs/data/amazon_books_small.yaml
```

```powershell
py -3.12 main_build_samples.py --config configs/data/amazon_electronics_small.yaml
```

#### 3. Run valid/test inference

```powershell
py -3.12 main_infer.py `
  --config configs/exp/books_small_deepseek.yaml `
  --input_path data/processed/amazon_books_small/valid.jsonl `
  --output_path outputs/books_small_deepseek/predictions/valid_raw.jsonl `
  --split_name valid `
  --max_samples 100 `
  --overwrite
```

```powershell
py -3.12 main_infer.py `
  --config configs/exp/books_small_deepseek.yaml `
  --input_path data/processed/amazon_books_small/test.jsonl `
  --output_path outputs/books_small_deepseek/predictions/test_raw.jsonl `
  --split_name test `
  --max_samples 100 `
  --overwrite
```

```powershell
py -3.12 main_infer.py `
  --config configs/exp/electronics_small_deepseek.yaml `
  --input_path data/processed/amazon_electronics_small/valid.jsonl `
  --output_path outputs/electronics_small_deepseek/predictions/valid_raw.jsonl `
  --split_name valid `
  --max_samples 100 `
  --overwrite
```

```powershell
py -3.12 main_infer.py `
  --config configs/exp/electronics_small_deepseek.yaml `
  --input_path data/processed/amazon_electronics_small/test.jsonl `
  --output_path outputs/electronics_small_deepseek/predictions/test_raw.jsonl `
  --split_name test `
  --max_samples 100 `
  --overwrite
```

#### 4. Run evaluation, calibration, and reranking

```powershell
py -3.12 main_eval.py --exp_name books_small_deepseek --input_path outputs/books_small_deepseek/predictions/test_raw.jsonl
py -3.12 main_calibrate.py --exp_name books_small_deepseek --valid_path outputs/books_small_deepseek/predictions/valid_raw.jsonl --test_path outputs/books_small_deepseek/predictions/test_raw.jsonl --method isotonic
py -3.12 main_rerank.py --exp_name books_small_deepseek --input_path outputs/books_small_deepseek/calibrated/test_calibrated.jsonl --lambda_penalty 0.5
```

```powershell
py -3.12 main_eval.py --exp_name electronics_small_deepseek --input_path outputs/electronics_small_deepseek/predictions/test_raw.jsonl
py -3.12 main_calibrate.py --exp_name electronics_small_deepseek --valid_path outputs/electronics_small_deepseek/predictions/valid_raw.jsonl --test_path outputs/electronics_small_deepseek/predictions/test_raw.jsonl --method isotonic
py -3.12 main_rerank.py --exp_name electronics_small_deepseek --input_path outputs/electronics_small_deepseek/calibrated/test_calibrated.jsonl --lambda_penalty 0.5
```

### Aggregate Results

To regenerate the current summary layer in one command:

```powershell
py -3.12 main_aggregate_all.py --output_root outputs
```

This entry point runs:

- `src/analysis/aggregate_domain_results.py`
- `src/analysis/aggregate_model_results.py`
- `src/analysis/aggregate_estimator_results.py`
- `src/analysis/robustness_summary.py`

If you only want the domain-level rerank/calibration summary, you can still run:

```powershell
py -3.12 src\analysis\aggregate_domain_results.py --output_root outputs
```

For a more complete experiment map, see:

- [docs/experiments.md](docs/experiments.md)
- [docs/tables.md](docs/tables.md)

### Robustness Baseline

The robustness line now extends beyond a single-model baseline. On `Beauty`, the repository currently supports multi-level noisy evaluation for:

- `beauty_deepseek`
- `beauty_glm`
- `beauty_qwen`
- `beauty_kimi`
- `beauty_doubao`

with `noise_level = 0.1 / 0.2 / 0.3` summarized in:

- `outputs/summary/beauty_robustness_curve_brief.csv`
- `outputs/summary/robustness_curve_results.csv`

The original Week2 entry point remains `Beauty + DeepSeek + noisy`, but the current paper-facing robustness layer is already a five-model Beauty comparison.

Generate noisy pointwise data:

```powershell
py -3.12 main_generate_noisy.py `
  --input_path data/processed/amazon_beauty/test.jsonl `
  --output_path data/processed/amazon_beauty_noisy/test.jsonl `
  --history_drop_prob 0.2 `
  --text_noise_prob 0.5 `
  --label_flip_prob 0.0
```

Run the noisy experiment:

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek_noisy.yaml `
  --input_path data/processed/amazon_beauty_noisy/test.jsonl `
  --output_path outputs/beauty_deepseek_noisy/predictions/test_raw.jsonl `
  --split_name test `
  --max_samples 100 `
  --overwrite
```

Then evaluate clean vs noisy:

```powershell
py -3.12 main_robustness.py --clean_exp beauty_deepseek --noisy_exp beauty_deepseek_noisy
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

Under `outputs/summary/`, the repository also maintains:

- `rerank_ablation.csv`: unified cross-domain / cross-lambda summary table
- `weekly_summary.csv`: compact view over diagnosis, calibration, and reranking metrics
- `final_results.csv`: consolidated cross-model, cross-domain result table with explicit `domain` and `lambda` columns
- `model_results.csv`: cross-domain / cross-model summary table
- `domain_model_summary.csv`: grouped model comparison per domain
- `estimator_results.csv`: multi-estimator comparison table
- `beauty_estimator_results.csv`: Beauty-focused estimator table for the main Day4 comparison
- `robustness_results.csv`: clean/noisy robustness summary rows
- `robustness_brief.csv`: compact robustness table for reporting

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

Current experiments are best understood as method-grounding and pipeline validation. Week1 already covers:

- Beauty as the main full-domain experiment
- Movies-small as the first cross-domain validation subset
- Books-small and Electronics-small as additional cross-domain validation subsets

Week2 and the current Week3 Beauty-first writing phase have already extended this base substantially:

- `DeepSeek`, `Qwen`, `GLM`, `Kimi`, and `Doubao` are all connected to the same inference / evaluation / calibration / reranking pipeline
- the current summary layer supports a `5 models x 4 domains` comparison setting
- multi-estimator comparison is available through verbalized, calibrated, consistency-based, and fused uncertainty definitions
- `Beauty` now has five-model clean comparisons, five-model estimator comparisons, and five-model multi-level robustness curves
- `Beauty` robustness is no longer only a `DeepSeek` story; `GLM`, `Qwen`, `Kimi`, and `Doubao` now follow the same `noise_level = 0.1 / 0.2 / 0.3` curve setup
- `Beauty` also includes consistency sensitivity and fused-alpha supporting analyses for paper writing

The current natural next step is not to add more raw experiments immediately, but to use the summary layer and docs layer to support paper writing and later larger-scale reruns.

## Notes

- Current small-scale runs are useful for validating methodology, not for claiming final large-scale empirical conclusions.
- The first reranking variant is intentionally conservative and interpretable.
- The codebase is structured so that new uncertainty estimators can be added without rewriting the evaluation chain.

## License

This project is released under the terms of the [LICENSE](LICENSE).
