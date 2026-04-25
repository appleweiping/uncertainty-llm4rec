# Day10 Plain Vs Evidence List Recommendation Report

## 1. Motivation

Day9 is candidate-level relevance posterior. Day10 tests whether the same idea transfers to list-level closed-catalog recommendation decisions.

## 2. Plain Baseline

The plain baseline uses the same user histories and candidate pools but forbids evidence fields. It only returns ranked candidate_item_id values and short reasons, so it is the no-scheme-four list recommendation control.

## 3. Evidence-Guided Method

The evidence-guided setting returns the same ranked list plus relevance_probability, positive/negative evidence, ambiguity, missing_information, and global_uncertainty. evidence_risk is derived after parsing.

## 4. Fair Comparison Setup

Both settings use the same `200` Beauty ranking users, identical candidate pools, the same DeepSeek backend, and the same top-K/evaluation code. The only intended difference is whether scheme-four evidence decomposition is available.

## 5. Evaluation

Plain direct list reaches HR@10 `1.0000`, NDCG@10 `0.6211`, and MRR@10 `0.4996`. Evidence-guided list reaches HR@10 `1.0000`, NDCG@10 `0.6311`, and MRR@10 `0.5135`. Evidence vs plain relative NDCG change is `0.0161` and relative MRR change is `0.0279`.

## 6. Bridge To Day9

The bridge table compares Day9 pointwise calibrated relevance, Day9 decoupled pointwise rerank, Day10 plain direct list generation, Day10 evidence-guided list generation, and Day10 evidence-guided list plus evidence-risk rerank on the same users and candidate pools.

## 7. Findings

The best method in this smoke test is `evidence_guided_list_risk_rerank` with NDCG@10 `0.6314` and MRR@10 `0.5138`. If evidence-guided generation does not dominate plain directly, the result should be read as evidence decomposition helping mainly through calibrated/risk-aware decision modules rather than automatically improving the first-pass list prompt.

## 8. Limitations

This is a 100/200-user smoke test, not full Beauty, not external SOTA, and not Qwen-LoRA. HR@10 is less informative because each Beauty candidate pool is small; NDCG/MRR are the primary list-order signals.

## 9. Next Step

If this smoke test is stable, the next Day10-full run should keep the same plain-vs-evidence control and then decide whether Qwen-LoRA should distill the evidence list schema, the Day9 pointwise schema, or both.
