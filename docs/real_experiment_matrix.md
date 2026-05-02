# Real Experiment Matrix

This matrix defines planned real experiment groups. It contains no completed
results. This repository is TRUCE-Rec unless the user states otherwise. All
entries below are planning targets: smoke/mock outputs are not paper evidence,
and pilot/API diagnostics are not paper conclusions unless a later approved
protocol promotes them with complete artifacts.

## Group A: Traditional baselines

| Experiment | Config path | Dataset | Method | Candidate protocol | Metrics | Output artifacts | Priority | Estimated compute cost | API key | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | `configs/experiments/real_main_movielens_template.yaml` | MovieLens TBD | `random` | full or shared sampled | shared evaluator | run dir + metrics | low | low | no | no |
| Popularity | `configs/experiments/real_main_movielens_template.yaml` | MovieLens/Amazon TBD | `popularity` | full or shared sampled | shared evaluator | run dir + metrics | high | low | no | no |
| BM25 | `configs/experiments/real_main_movielens_template.yaml` | MovieLens/Amazon TBD | `bm25` | full or shared sampled | shared evaluator | run dir + metrics | high | low | no | no |
| MF | `configs/experiments/real_main_movielens_template.yaml` | MovieLens/Amazon TBD | `mf` | full or shared sampled | shared evaluator | run dir + metrics | medium | low-medium | no | optional |
| Sequential Markov | `configs/experiments/real_main_amazon_template.yaml` | Amazon TBD | `sequential_markov` | shared sampled if large | shared evaluator | run dir + metrics | high | low | no | no |
| SASRec interface | TBD real config | Amazon TBD | `sasrec_interface` or real SASRec | shared sampled | shared evaluator | checkpoint + run dir | later | medium-high | no | likely |

## Group B: LLM baselines

| Experiment | Config path | Dataset | Method | Candidate protocol | Metrics | Output artifacts | Priority | Estimated compute cost | API key | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLM generative | `configs/experiments/real_observation_template.yaml` | MovieLens/Amazon TBD | `llm_generative` | visible non-target candidates | shared + confidence | raw outputs + run dir | high | API/model dependent | maybe | maybe |
| LLM rerank | `configs/experiments/real_llm_api_template.yaml` | MovieLens/Amazon TBD | `llm_rerank` | shared candidate set | shared + cost | raw outputs + run dir | high | API/model dependent | yes for API | no |
| LLM confidence observation | `configs/experiments/real_observation_template.yaml` | MovieLens/Amazon TBD | `llm_confidence_observation` | shared candidate set | confidence/calibration | raw outputs + analysis CSVs | high | API/model dependent | maybe | maybe |
| API model variant | `configs/experiments/real_llm_api_template.yaml` | TBD | OpenAI-compatible | shared candidate set | shared + cost | API cache/raw outputs | pilot | budgeted | yes | no |
| HF local variant | `configs/experiments/real_observation_template.yaml` | TBD | HF local provider | shared candidate set | shared + latency | local raw outputs | later | hardware dependent | no | likely |

## Group C: OursMethod

| Experiment | Config path | Dataset | Method | Candidate protocol | Metrics | Output artifacts | Priority | Estimated compute cost | API key | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ours full | `configs/experiments/real_ours_method_template.yaml` | MovieLens then Amazon TBD | `ours_uncertainty_guided` | shared candidate set | all shared + policy counts | predictions + metrics + cost | high | provider dependent | maybe | maybe |
| Ours fallback-only | `configs/experiments/real_ablation_template.yaml` | same as full | `ours_fallback_only` | same as full | all shared | run dir | high | low | no | no |
| Ours w/o uncertainty | `configs/experiments/real_ablation_template.yaml` | same as full | `ours_ablation_no_uncertainty` | same as full | all shared | run dir | high | provider dependent | maybe | maybe |
| Ours w/o grounding | `configs/experiments/real_ablation_template.yaml` | same as full | `ours_ablation_no_grounding` | same as full | validity/hallucination focus | run dir | high | provider dependent | maybe | maybe |
| Ours w/o candidate-normalized confidence | `configs/experiments/real_ablation_template.yaml` | same as full | `ours_ablation_no_candidate_normalized_confidence` | same as full | calibration focus | run dir | high | provider dependent | maybe | maybe |
| Ours w/o popularity adjustment | `configs/experiments/real_ablation_template.yaml` | same as full | `ours_ablation_no_popularity_adjustment` | same as full | popularity/long-tail focus | run dir | high | provider dependent | maybe | maybe |
| Ours w/o echo guard | `configs/experiments/real_ablation_template.yaml` | same as full | `ours_ablation_no_echo_guard` | same as full | diversity/novelty focus | run dir | high | provider dependent | maybe | maybe |

## Group D: Observation-only analysis

| Experiment | Config path | Dataset | Method | Candidate protocol | Metrics | Output artifacts | Priority | Estimated compute cost | API key | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Verbalized confidence | `configs/experiments/real_observation_template.yaml` | TBD | confidence observation | shared | calibration | reliability CSVs | high | provider dependent | maybe | maybe |
| Yes/no verification | `configs/experiments/real_observation_template.yaml` | TBD | observation signal | shared | confidence vs correctness | raw + parsed outputs | high | provider dependent | maybe | maybe |
| Candidate-normalized confidence | `configs/experiments/real_observation_template.yaml` | TBD | observation signal | shared | calibration/risk | normalized outputs | high | provider dependent | maybe | maybe |
| Multi-sample consistency | TBD later | TBD | later implementation | shared | entropy/agreement | multi-sample outputs | later | high | maybe | maybe |
| Popularity-stratified calibration | `configs/experiments/real_observation_template.yaml` | Amazon TBD | analysis | shared | ECE by bucket | confidence bucket CSVs | high | analysis only after runs | no | no |
| Long-tail under-confidence | `configs/experiments/real_observation_template.yaml` | Amazon TBD | analysis | shared | tail confidence gaps | long-tail CSVs | high | analysis only after runs | no | no |
| High-confidence hallucination | `configs/experiments/real_observation_template.yaml` | TBD | analysis | shared | high-conf wrong/hallucination | failure cases | high | analysis only after runs | no | no |
| Confidence-weighted diversity/novelty | `configs/experiments/real_observation_template.yaml` | TBD | analysis | shared | diversity/novelty | table/plot data | medium | analysis only after runs | no | no |
