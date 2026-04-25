# Day9 Full Beauty Relevance Evidence Report

## 1. Day8 Recap

Day8 scaled candidate relevance evidence from the Day7 smoke test to a medium Beauty sample. It showed parse_success of 1.0, persistent raw relevance miscalibration, strong ECE/Brier gains after valid-set calibration, and nonzero decoupled rerank changes. Day9 therefore moves to the final full-Beauty relevance-evidence shape before deciding what to distill into a local Qwen-LoRA generator.

## 2. Full Setup

The experiment `beauty_deepseek_relevance_evidence_full` uses DeepSeek API evidence generation with `prompts/candidate_relevance_evidence.txt`, concurrent/resumable inference, and no LoRA or external baseline. The full valid split contains `5838` rows and the full test split contains `5838` rows. Calibration is fit only on valid and applied to test.

## 3. Parse And Field Diagnostics

Valid parse_success is `1.0000` and test parse_success is `1.0000`. On test, relevance_probability has mean `0.4270`, std `0.2174`, and near_one_rate `0.0003`. evidence_risk has mean `0.4303` and std `0.1230`. This checks whether the relevance field collapses toward extreme self-confidence; the detailed distribution is in the field diagnostics table.

## 4. Calibration

Raw relevance has ECE `0.2604`, Brier `0.2318`, and AUROC `0.6007`. Valid-set calibrated relevance has ECE `0.0095`, Brier `0.1347`, and AUROC `0.5986`. Minimal evidence posterior has ECE `0.0096`, Brier `0.1349`, and AUROC `0.6010`. Full evidence posterior has ECE `0.0116`, Brier `0.1348`, and AUROC `0.6038`.

## 5. Rerank

The best full rerank row is setting `R-B`, normalization `zscore`, lambda `0.1`, base score `relevance_probability`, and uncertainty `evidence_risk`. It reaches HR@10 `1.0000`, NDCG@10 `0.6143`, MRR@10 `0.4910`, rank_change_rate `0.2206`, and top10_order_change_rate `0.4933`.

## 6. Comparison To Day6 Yes/No Full

Day6 and Day9 should not be collapsed into one leaderboard row. Day6 is the full Beauty yes/no confidence-repair branch and proved that decoupling relevance and risk fixes the monotonic no-op. Day9 is the final candidate relevance scoring shape: relevance_probability is the main recommendation score, while evidence_risk is a separate decision risk.

## 7. Limitation

This is still a DeepSeek API evidence generator, not a local Qwen-LoRA model and not an external SOTA comparison. The current result establishes the full-Beauty feasibility and behavior of the relevance-evidence formulation before model compression or broader baseline work.

## 8. Day10 Recommendation

The full rows preserve the Day8 pattern: parse_success remains stable, raw relevance remains miscalibrated, valid-set calibration/evidence posterior sharply improve ECE and Brier, and decoupled reranking produces nonzero rank changes. Day10 should therefore prepare the Qwen-LoRA relevance distillation taskbook and an external baseline plug-in plan. The distillation target should use candidate relevance evidence, not the older yes/no confidence wording.
