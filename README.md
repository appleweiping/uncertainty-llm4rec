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

## Current Project Positioning (April 2026)

The project now has two closely related, but distinct, layers.

The first layer is the teacher-requested execution line. The original practical goal was to move the uncertainty-aware recommendation pipeline away from heavy dependence on official API calls and toward a local, server-runnable `8B + LoRA` small-model setup. In the current roadmap, this corresponds to `week7.8` and `week7.9`. The purpose of this line is not to invent a new method family first, but to make the original uncertainty pipeline run at larger scale with a local model:

- `week7.8`: replay the true `week1-week4` uncertainty pipeline on full-domain local-`v2`
- `week7.9`: tighten fairness, baseline-confidence alignment, and metric/protocol auditing before formal comparison

The second layer is the later paper-facing research line. Once the local `8B + LoRA` execution path became stable, the project naturally evolved into a deeper question: if a stronger teacher provides recommendation corrections together with uncertainty signals, when should a smaller model learn from them, how strongly should it learn, and how should it avoid learning harmful corrections? This is the line that leads into `week8` and `week9`:

- `week8`: formal outer comparison and same-schema baseline/proxy comparison
- `week9`: system integration, where teacher reliability, correction selection, preference construction, and student adaptation are collected into one framework

This distinction matters because the repository should not be read as "a LoRA fine-tuning project" alone. The local `8B + LoRA` student is the execution carrier. The central research question is about uncertainty-aware recommendation and, later, uncertainty-aware conditional student learning.

## Current Weekly Roadmap

The near-term roadmap is intentionally layered rather than flat:

1. `week7.8` finishes the teacher-requested local execution line on full-domain data.
2. `week7.9` builds the fairness bridge so that confidence-related claims are not evaluated only on our own mainline.
3. `week8` performs formal outer comparison between direct baselines, structured-risk baselines, SRPD variants, and literature-motivated proxies under a controlled schema.
4. `week9` turns the accumulated mechanisms into a system-level narrative rather than leaving them as a list of variants.

Another important clarification is that `pointwise_yesno` is still present, but no longer acts as the project's final task formulation. It is now the diagnosis and calibration layer. The main recommendation decision layer has been upgraded to candidate ranking and reranking, while pairwise preference serves as a mechanism/supporting layer.

## Current Research Direction

The repository is no longer a single-task pointwise project. It is being expanded into a unified multi-decision framework in which pointwise remains the diagnosis layer, ranking is the main decision layer, and pairwise acts as the mechanism/supporting layer:

- `pointwise_yesno` remains the diagnostic layer for uncertainty elicitation and calibration
- `pairwise_preference` is being introduced as the local preference-mechanism layer
- `candidate_ranking` is being introduced as the primary listwise decision layer

The current transition keeps the old pointwise chain intact while adding new task-specific inputs, configs, and later task-specific inference/evaluation paths.

To support this transition, the repository now also defines task-specific language interfaces:

- `prompts/candidate_ranking.txt` and `prompts/pairwise_preference.txt` provide explicit structured-output templates for the two new decision tasks
- `src/llm/prompt_builder.py` now constructs pointwise, ranking, and pairwise prompts from a shared interface layer
- `src/llm/parser.py` now includes task-specific parsing paths for candidate ranking and pairwise preference outputs, including fenced JSON, partial JSON, and lightweight free-form recovery cases

This keeps the multi-task expansion aligned with the old project principle: prompt design and parser robustness are part of the method boundary, not an afterthought.

The inference layer is now also being separated by task boundary rather than collapsed into a single entry point:

- `main_infer.py` remains the pointwise diagnostic entry
- `main_rank.py` serves the candidate ranking decision path
- `main_pairwise.py` serves the pairwise preference path

This preserves pointwise clarity while allowing ranking and pairwise outputs to land as first-class prediction artifacts rather than side experiments.

The evaluation layer is now moving in the same direction:

- `main_eval.py` remains responsible for pointwise diagnosis, calibration quality, and confidence-related bias views
- `main_eval_rank.py` evaluates candidate ranking outputs in the recommendation metric space
- `main_eval_pairwise.py` evaluates pairwise preference outputs in the local preference-consistency space

This keeps the multi-task expansion methodologically honest: different decision granularities now produce different prediction files and are judged by different, task-appropriate metrics.

The repository now also begins to aggregate these task-specific results back into a single paper-facing comparison layer:

- `src/analysis/aggregate_multitask_results.py` collects pointwise, pairwise, and ranking summaries under one schema
- `main_compare_multitask.py` produces a unified cross-task comparison table and a compact narrative summary

This matters because the project is no longer asking only whether confidence is calibrated in isolation, but also how uncertainty behaves across different decision granularities.

The next method step is to let uncertainty move from the diagnostic layer into the decision layer itself:

- `src/methods/uncertainty_ranker.py` maps listwise ranking outputs into candidate-level scores and injects explicit uncertainty penalties
- `main_rank_rerank.py` applies pointwise-derived uncertainty signals to candidate ranking predictions and evaluates the reranked list under the same ranking metrics

This transition is important because it changes uncertainty from a pointwise observation variable into a listwise control variable that can directly alter resource allocation across candidates.

The current ranking decision layer intentionally keeps the uncertainty control family simple and paper-friendly:

- a plain linear penalty remains the baseline transfer rule
- a coverage-aware linear penalty softens the penalty when aligned uncertainty support is partial
- a top-k gated penalty concentrates uncertainty intervention on the most decision-critical prefix of the list

These variants are not meant to be the final method family. They are the first controlled compare set for studying how pointwise-derived uncertainty should enter listwise recommendation decisions under imperfect alignment.

The current week-six method line now also includes a more structured decision family aimed at recovering utility rather than only validating transfer:

- `nonlinear_structured_risk_rerank` combines position gating, uncertainty thresholding, non-linear risk amplification, coverage-aware weakening, and a small protection bonus for high-relevance low-risk candidates
- `local_margin_swap_rerank` applies a local swap correction when nearby candidates have small normalized relevance gaps but clear uncertainty gaps
- `structured_risk_plus_local_margin_swap_rerank` stacks the two ideas so that global risk shaping and local order repair can be evaluated together under one interface

This extension is driven by an empirical failure mode rather than formula inflation: simple uncertainty penalties already show that uncertainty can enter ranking decisions, but they do not yet recover or improve ranking utility. The structured-risk line is therefore designed to be more selective, more position-aware, and more honest about partial uncertainty coverage.

The mechanism layer is now being pushed one step further rather than left as a diagnostic side branch:

- `src/methods/uncertainty_pairwise_aggregator.py` turns pairwise preference outputs into event-level candidate scores through uncertainty-weighted preference aggregation
- `main_pairwise_rank.py` evaluates whether pairwise preference can act as a ranking-construction path instead of stopping at pairwise accuracy

This compare is intentionally disciplined. When pairwise predictions cover only part of the ranking set, the repository evaluates pairwise-to-rank against direct ranking and retained ranking families on the event-overlap subset rather than mixing unmatched events into a single headline table. The goal is to keep the mechanism evidence useful without overstating how complete it already is.

The current summary layer now also begins to compare uncertainty sources across decision granularities instead of only within the old pointwise diagnostic view:

- `main_uncertainty_compare_multitask.py` aligns verbalized raw, calibrated, self-consistency, and fused uncertainty under one multitask evaluation entry
- `outputs/summary/week6_day3_estimator_compare.csv` places pointwise diagnosis, candidate-ranking rerank families, and pairwise-to-rank results under one paper-facing schema
- `outputs/summary/week6_day3_pairwise_coverage_compare.csv` records pairwise support rate, pair coverage, and overlap-vs-expanded ranking results so that mechanism gains can be interpreted together with coverage limits

This matters because the project is no longer asking only which uncertainty source calibrates best in isolation. It is now asking a stronger question: which uncertainty source remains useful after it is pushed through different decision formulations, and whether pairwise gains still hold once limited support is expanded with explicit direct-ranking fallback rather than hidden by metric mismatch.

The project now also starts to maintain a same-task baseline layer for the multi-decision setting instead of relying only on within-method comparisons:

- `src/methods/baseline_ranker_multitask.py` aligns pointwise calibrated-confidence baselines, direct candidate-ranking baselines, and plain pairwise aggregation baselines under one compare schema
- `main_baseline_multitask.py` produces `outputs/summary/week6_day4_decision_baseline_compare.csv`
- retained complex families remain visible in that table rather than disappearing once they fail to become the current best default line

This baseline layer is intentionally narrower than the later literature-aligned benchmark stage. Its role is to make the Part5 method story experimentally honest first: each decision layer should have a same-task, non-uncertainty reference before stronger external baselines are brought in.

The Part5 branch can now be finalized into a compact paper-facing evidence bundle:

- `main_compare_multitask.py --finalize_part5` converts the same-task baseline matrix into `outputs/summary/part5_multitask_final_results.csv`
- `outputs/summary/part5_multitask_final_summary.md` records the current interpretation of pointwise, candidate ranking, and pairwise-to-rank under the unified Part5 setting

This finalization step is deliberately conservative. It identifies the current ranking family, keeps pairwise-to-rank as a coverage-limited mechanism line, and preserves complex retained variants without presenting them as the default method.

The immediate research handoff is therefore not another round of formula expansion. The current Part5 branch should be treated as a closed method loop that still needs stronger evidence: paper-facing figures, better pairwise coverage, controlled medium-scale execution, four-domain DeepSeek validation on compact samples, and later literature-aligned baselines. Official API models remain useful as small external observation windows, but the main evidence path should move toward reproducible batch execution.

The Part5 evidence layer now has a dedicated artifact entry point:

- `main_part5_artifacts.py --refresh_part5_final --build_figures --build_pairwise_coverage` rebuilds the final Part5 table, paper-facing figure pack, and pairwise coverage evidence pack from existing summaries.
- Part5 figures are written under `outputs/summary/figures/part5/`, and their plot-source tables are written under `outputs/summary/tables/part5/`.
- `outputs/summary/part5_figure_pack.md` records how the figures support the current paper narrative without opening new ranking families.

The cross-domain handoff is also now represented as a compact four-domain DeepSeek matrix:

- `configs/exp/*_deepseek_pointwise.yaml` and `configs/exp/*_deepseek_rank.yaml` define compact Movies, Beauty, Books, and Electronics runs.
- `src/analysis/aggregate_cross_domain_minimal_results.py` writes the four-domain status and compare tables under `outputs/summary/`.
- `main_literature_baseline.py` and `main_batch_run.py` provide the first minimal baseline and registry hooks for the next larger-scale stage.

The current Part5 compact evidence version is now complete under the 100-sample real-API scope. Week5 supplies the multitask data, prompt/parser, inference, evaluation, and summary base; Week6 supplies the ranking decision family, pairwise-to-rank mechanism line, estimator compare, same-task baseline, final Part5 table, and final summary. The magic supplement adds paper-facing figures, pairwise coverage evidence, the four-domain DeepSeek compact matrix, baseline/registry handoff hooks, and the final structured-risk cross-domain landing.

The default Part5 ranking line is `nonlinear_structured_risk_rerank`, treated as the current best structured-risk family. `local_margin_swap_rerank` and `structured_risk_plus_local_margin_swap_rerank` remain retained exploratory families: they stay visible in code, tables, figures, and documentation, but they are not the default main experimental line. Pairwise-to-rank is currently interpreted as mechanism-layer evidence rather than a replacement for candidate ranking; its coverage has been upgraded in the compact Beauty setting, while the current evidence still remains intentionally bounded to the 100-sample scope.

The final Part5 evidence layer is organized through these main entrances:

- `main_compare_multitask.py --finalize_part5` refreshes the unified Part5 results and summary.
- `main_part5_artifacts.py --build_figures --build_pairwise_coverage` rebuilds paper-facing figures and coverage evidence.
- `main_rank_rerank.py` runs the current structured-risk ranking family on candidate-ranking outputs.
- `src/analysis/aggregate_cross_domain_structured_risk_results.py` consolidates four-domain direct-vs-structured-risk results.
- `main_literature_baseline.py` runs the compact task-aligned baseline group.
- `src/analysis/build_part5_consolidated_tables.py` builds the paper-ready consolidated tables.
- `part5_artifact_map.md` records the main tables, figures, notes, and reports for this stage.

Week7 begins the execution-layer upgrade for the next validation phase. The repository now distinguishes API backends as observation/comparison channels and the server-side Hugging Face backend as the main experiment path. The selected local main model is now Qwen3-8B, configured through `configs/model/qwen3_8b_local.yaml`; the currently verified server-local ModelScope path is `/home/ajifang/models/Qwen/Qwen3-8B`, loaded with `local_files_only` enabled. The backend implementation is `src/llm/local_hf_backend.py`, with `main_backend_check.py` providing a minimal schema and loading check before larger runs. Qwen3 output compatibility is handled at the execution layer through final-answer prompt constraints, `enable_thinking: false` where the chat template supports it, and parser-side guards for thinking blocks and confidence normalization shared by pointwise, candidate ranking, and pairwise tasks. This is an execution-backbone change only: the structured-risk current best family remains the main ranking line, pairwise remains the mechanism line, and retained ranking families keep their previous roles. Server credentials and temporary access links are intentionally kept outside the repository.

Week7 also now has a structured batch and registry layer. `main_batch_run.py` reads `configs/batch/week7_local_scale.yaml`, builds task-specific commands for pointwise, candidate ranking, and pairwise smoke runs, and writes `outputs/summary/week7_day2_batch_status.csv`. The default mode is dry-run/registration; real server execution requires `--run`, while `--only_failed` supports targeted recovery from a previous registry. The related workflow notes live in `docs/week7_day2_batch_workflow.md` and `docs/week7_server_execution.md`.

The first Part6 literature-aligned baseline group is now represented through `main_run_literature_baselines.py` and `configs/baseline/week7_literature_baselines.yaml`. It keeps the scope intentionally small but strict: ranking baselines and pairwise preference baselines run on the same Beauty candidate and pairwise samples, produce task-native prediction files, and summarize into `outputs/summary/week7_day3_literature_baseline_summary.csv`. The schema already records `model_family` and `adapter_path`, so the current base-only Qwen3 local path can later be extended to adapter-based comparisons without redefining the baseline table.

The baseline evidence layer now also has a unified matrix entrance. `main_compare_baselines.py` reads the uncertainty-source comparison, same-task decision baseline table, and first literature-aligned baseline summary, then writes `outputs/summary/week7_day4_baseline_matrix.csv`. The companion note `docs/week7_day4_baseline_system.md` records the intended roles of official API observation models, the server-side local-HF main model group, and the reserved LoRA-adapted group without turning README into a runbook.

The Week7 medium-scale handoff is represented through `configs/batch/week7_medium_scale.yaml` and `main_compare_week7_medium_scale.py`. The batch list registers Beauty pointwise, direct candidate ranking, pairwise, and the structured-risk current best rerank line under the same server-local Qwen3 identity. The server run plan lives in `docs/week7_day5_server_run_plan.md`; it fixes the base-only first route, keeps LoRA as a later adapter-only extension, and reserves vLLM for a future throughput stage.

This handoff is no longer only a dry-run shell. The current server-local Qwen3 medium execution has already completed for Beauty at 300 samples per task, followed by pointwise isotonic calibration, structured-risk reranking, and task-native evaluation. The resulting summary is written to `outputs/summary/week7_day5_medium_scale_summary.csv`. In that setting, the direct ranking line reaches `NDCG@10=0.6244063031442281` and `MRR=0.5039761904761905`, while the current best structured-risk rerank line reaches `NDCG@10=0.624482549323477` and `MRR=0.5040555555555557`, with full rerank coverage and lower head exposure than the direct ranking medium line. Pairwise medium evaluation is also materialized in the same summary with `pairwise_accuracy=0.61`. This means Week7 has already crossed the boundary from backend scaffolding into real medium-scale server evidence under the Qwen3 local execution backbone.

Before continuing server-side Week7 runs, the server data layer should be aligned with the local processed pony data rather than the old partial `data/` upload. The upload manifest in `docs/week7_server_data_upload_manifest.md` defines the minimal Beauty processed directory for immediate Qwen3 smoke / medium-scale runs and the recommended compact four-domain processed upload for the next validation stage.

The next engineering step is now formalized as Week7.5 rather than a premature jump into Week8. This stage does not reopen the ranking-family search and does not expand into a four-domain matrix yet. Instead, it adds a training-stage center object: a Beauty-only, candidate-ranking-only `Qwen3-8B + LoRA` framework skeleton driven by `configs/lora/qwen3_rank_beauty_framework_v1.yaml`, `main_lora_train_rank.py`, `main_eval_lora_rank.py`, and `main_compare_framework.py`. The role split remains unchanged: pointwise stays the diagnosis/calibration layer, pairwise stays the mechanism line, and candidate ranking is the only first-round trainable task. In that setup, the current structured-risk rerank line is preserved explicitly as the strongest hand-crafted baseline that the future trainable framework must face rather than silently replace.

Week7.5 Day2 moves this framework from a planning skeleton into a launchable training path. `main_lora_train_rank.py` now supports `--startup_check` so that the Beauty ranking train/valid/eval paths, prompt path, output directories, and supervised-example construction can be validated before any LoRA job is launched. The resulting startup report is written to `outputs/summary/week7_5_startup_check.json`, while split-level data summaries are written to `outputs/summary/week7_5_dataset_preview.csv`, and training state is tracked in `outputs/summary/week7_5_train_status.csv`. A companion workflow note lives in `docs/week7_5_lora_training_workflow.md`, and `configs/batch/week7_5_rank_train.yaml` records the three-step server handoff path: startup check, training, and post-training framework evaluation. This keeps the project in the intended Week7.5 shape: candidate ranking is the only trainable task, structured risk remains the strongest hand-crafted baseline, and pointwise/pairwise continue as supporting layers rather than being collapsed into a premature joint-training loop.

Week7.5 Day3 further tightens this into a closure-ready local engineering path. The training side now updates a framework run manifest under `outputs/beauty_qwen3_rank_framework_v1/framework_run_manifest.json`, the evaluation side writes `framework_eval_summary.csv` next to the standard ranking metrics, and the compare side refreshes both `outputs/summary/week7_5_framework_compare.csv` and `outputs/summary/week7_5_framework_compare.md`. In other words, the project now has a single Beauty-domain closure path for the trainable framework: `startup_check -> train -> eval -> compare`. This is still not a claim that the LoRA framework has already been trained locally; it is an engineering guarantee that once the server run happens, the resulting adapter, framework metrics, and compare evidence can land in the project without another round of local plumbing.

Week7.5 Day4 makes this closure path baseline-aware rather than framework-local only. The framework compare schema is now tightened around explicit role labels such as `same_task_reference`, `strongest_handcrafted_baseline`, `literature_or_task_aligned_reference`, and `trainable_framework_main_candidate`. `structured risk current best family` is fixed as the strongest hand-crafted baseline that the future LoRA framework must face, not silently replaced. At the same time, `main_compare_framework.py` now also builds a bridge matrix under `outputs/summary/week7_5_baseline_matrix.csv` plus a companion markdown summary, so that once real server-side framework metrics arrive, they can enter the existing Week7 baseline matrix semantics without reopening the compare design.

Week7.5 Day5 turns the next bottleneck into an explicit execution contract instead of an informal warning. Ranking throughput is now treated as the primary speed problem to solve, but the optimization boundary is fixed: speed-up is not allowed to come from deleting pointwise diagnosis, pairwise mechanism evidence, calibration, or the three-layer baseline compare. The new entry `main_prepare_speed_upgrade.py` materializes this as `outputs/summary/week7_5_speed_upgrade_plan.csv` and `outputs/summary/week7_5_speed_upgrade.md`, covering tighter ranking output contracts, verification that `batch_generate` is actually used, reduced repeated model loads, shard/resume expectations, and a reserved transition path toward resident serving once the project grows beyond the Beauty single-domain medium stage. This keeps Week8 honest: the framework must be both compare-ready and execution-ready before the larger matrix opens.

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
|   |-- baseline/             # literature-aligned baseline configs
|   |-- task/                 # task-level builder and interface configs
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
|-- main_build_multitask_samples.py
|-- main_infer.py
|-- main_rank.py
|-- main_pairwise.py
|-- main_backend_check.py
|-- main_eval.py
|-- main_eval_rank.py
|-- main_eval_pairwise.py
|-- main_compare_multitask.py
|-- main_rank_rerank.py
|-- main_uncertainty_compare_multitask.py
|-- main_baseline_multitask.py
|-- main_literature_baseline.py
|-- main_run_literature_baselines.py
|-- main_compare_baselines.py
|-- main_compare_week7_medium_scale.py
|-- main_calibrate.py
|-- main_rerank.py
`-- main_uncertainty_compare.py
```

## Paper-Facing Summary Layer

For paper writing, the repository now maintains two result layers:

- experiment-complete summaries under `outputs/summary/`
- Beauty-centered paper-facing tables derived from those summaries

For the new multi-task branch, the summary layer also starts to maintain a cross-task comparison view:

- `outputs/summary/week5_multitask_summary.csv`
- `outputs/summary/week5_multitask_summary.md`
- `outputs/summary/week6_day3_estimator_compare.csv`
- `outputs/summary/week6_day3_pairwise_coverage_compare.csv`
- `outputs/summary/week6_day4_decision_baseline_compare.csv`
- `outputs/summary/part5_multitask_final_results.csv`
- `outputs/summary/part5_multitask_final_summary.md`
- `outputs/summary/part5_figure_pack.md`
- `outputs/summary/week6_magic7_4domain_deepseek_compare.csv`

These files are designed to answer a narrower but important question than the full paper tables: what role does uncertainty appear to play at pointwise, pairwise, and ranking decision granularities once the first multi-task pipeline is closed?

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

The repository is organized around four config layers:

- `configs/data/`: domain-specific preprocessing and sample-building settings
- `configs/task/`: task-specific sample-building and interface settings
- `configs/model/`: backend, model, connection, and generation settings
- `configs/exp/`: experiment-level inference settings such as `exp_name`, input path, output root, prompt path, and model config

This design keeps experiments reproducible and makes Week2 extensions easier. In practice:

- changing domains should mostly mean switching `configs/data/*.yaml`
- changing task formulations should mostly mean switching `configs/task/*.yaml`
- changing models should mostly mean switching `configs/model/*.yaml`
- changing an experiment run should mostly mean switching `configs/exp/*.yaml`

For the multi-task transition, `main_build_multitask_samples.py` derives:

- `ranking_valid.jsonl` and `ranking_test.jsonl`
- `pairwise_valid.jsonl` and `pairwise_test.jsonl`

from the existing pointwise evaluation splits, while preserving per-user alignment to the same positive target event.

The repository can now also land three task-specific prediction files under experiment directories:

- pointwise predictions in `predictions/test_raw.jsonl`
- ranking predictions in `predictions/rank_predictions.jsonl`
- pairwise predictions in `predictions/pairwise_predictions.jsonl`

This makes later multi-task evaluation and aggregation much cleaner because the diagnostic, preference, and ranking layers no longer share the same raw prediction schema.

The corresponding evaluation outputs are also beginning to separate by task:

- pointwise tables such as `diagnostic_metrics.csv`, `confidence_bins_accuracy.csv`, and exposure-oriented figures
- ranking tables such as `ranking_metrics.csv`
- pairwise tables such as `pairwise_metrics.csv` and `preference_confidence_bins.csv`

This separation is intentional: pointwise remains the uncertainty diagnosis layer, while ranking and pairwise are evaluated in their own task spaces before later cross-task comparison.

For candidate ranking, the repository also starts to support uncertainty-aware decision-time reranking:

- direct listwise predictions remain in `predictions/rank_predictions.jsonl`
- uncertainty-aware listwise outputs land in `reranked/rank_reranked.jsonl`
- corresponding comparison tables land in `tables/rerank_results.csv`

This extends the original uncertainty-aware reranking idea beyond pointwise candidate judgments and into the listwise decision layer.

Current model config files include:

- `configs/model/deepseek.yaml`: active backend used in Week1 experiments
- `configs/model/qwen.yaml`: Qwen API-compatible backend config
- `configs/model/glm.yaml`: GLM API-compatible backend config
- `configs/model/doubao.yaml`: Doubao API-compatible backend config
- `configs/model/kimi.yaml`: Kimi API-compatible backend config

Environment variables are read from model configs via `api_key_env`. Typical examples are:

- `DEEPSEEK_API_KEY`
- `QWEN_API_KEY`
- `DOUBAO_API_KEY`
- `KIMI_API_KEY`

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
- `week6_final_4domain_structured_risk_compare.csv`: four-domain direct ranking vs structured-risk current best comparison
- `week6_final_pairwise_coverage_upgrade.csv`: upgraded pairwise event coverage and overlap/expanded boundary table
- `week6_final_literature_baseline_compare.csv`: compact task-aligned and literature-aligned ranking baseline compare table
- `week7_day3_literature_baseline_summary.csv`: first Part6 ranking and pairwise literature-aligned baseline summary
- `week7_day4_baseline_matrix.csv`: unified uncertainty-source, decision-formulation, and literature-aligned baseline matrix
- `week7_day5_batch_status.csv` and `week7_day5_medium_scale_summary.csv`: server-local medium-scale status and consolidated summary after the real Beauty Qwen3 medium run
- `part5_single_domain_main_table.csv`, `part5_4domain_main_table.csv`, and `part5_pairwise_boundary_table.csv`: paper-ready consolidated Part5 table skeletons

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
- a compact Part5 multitask evidence version with pointwise diagnosis, pairwise mechanism evidence, candidate-ranking structured-risk reranking, four-domain DeepSeek compact replication, upgraded pairwise coverage, and paper-ready consolidated tables
- the first Week7 server-backend handoff: unified local-HF backend wiring, Qwen3-8B ModelScope-local model config, thinking-output cleanup, smoke experiment configs, server workflow docs, and a recoverable batch registry for local-HF smoke runs
- the first Part6 literature-aligned baseline handoff: compact ranking and pairwise baseline groups under the same candidate/pairwise samples, with base-only and future adapter identity represented in the summary schema
- a unified Part6 baseline matrix that places uncertainty-source, decision-formulation, and literature-aligned baselines under one schema before the server-side local-HF runs are scaled
- a Week7 medium-scale server handoff that registers base-only Qwen3 pointwise/ranking/pairwise runs and the structured-risk current best rerank line without reopening the ranking-family search

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

The repository now also carries a parallel cross-domain small-scope configuration family for `Books`, `Electronics`, and `Movies` under the local Qwen3 and SRPD routes. These configs keep pointwise, ranking, pairwise, teacher-data, and LoRA outputs on separate paths from the main Beauty artifacts so that cross-domain validation can be expanded without overwriting the current Beauty evidence line.

## Notes

- Current small-scale runs are useful for validating methodology, not for claiming final large-scale empirical conclusions.
- The first reranking variant is intentionally conservative and interpretable.
- The codebase is structured so that new uncertainty estimators can be added without rewriting the evaluation chain.

## License

This project is released under the terms of the [LICENSE](LICENSE).
