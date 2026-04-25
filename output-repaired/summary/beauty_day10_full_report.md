# Day10 Plain Vs Evidence List Recommendation Report

## 1. Motivation

Day9 is candidate-level relevance posterior. Day10 tests whether the same idea transfers to list-level closed-catalog recommendation decisions.

## 2. Plain Baseline

The plain baseline uses the same user histories and candidate pools but forbids evidence fields. It only returns ranked candidate_item_id values and short reasons, so it is the no-scheme-four list recommendation control.

## 3. Evidence-Guided Method

The evidence-guided setting returns the same ranked list plus relevance_probability, positive/negative evidence, ambiguity, missing_information, and global_uncertainty. evidence_risk is derived after parsing.

## 4. Fair Comparison Setup

Both settings use the same `973` Beauty ranking users, identical candidate pools, the same DeepSeek backend, and the same top-K/evaluation code. The only intended difference is whether scheme-four evidence decomposition is available.

## 5. Evaluation

Plain direct list reaches HR@10 `1.0000`, NDCG@10 `0.6260`, and MRR@10 `0.5065`. Evidence-guided list reaches HR@10 `1.0000`, NDCG@10 `0.6233`, and MRR@10 `0.5030`. Evidence vs plain relative NDCG change is `-0.0043` and relative MRR change is `-0.0068`.

## 6. Bridge To Day9

The bridge table compares Day9 pointwise calibrated relevance, Day9 decoupled pointwise rerank, Day10 plain direct list generation, Day10 evidence-guided list generation, and Day10 evidence-guided list plus evidence-risk rerank on the same users and candidate pools.

## 7. Findings

The best method in this full Beauty run is `plain_direct_list` with NDCG@10 `0.6260` and MRR@10 `0.5065`. If evidence-guided generation does not dominate plain directly, the result should be read as evidence decomposition helping mainly through calibrated/risk-aware decision modules rather than automatically improving the first-pass list prompt.

## 8. Limitations

This is full Beauty for the current closed-catalog candidate pools, but it is still not external SOTA and not Qwen-LoRA. HR@10 is less informative because each Beauty candidate pool is small; NDCG/MRR are the primary list-order signals.

## 9. Next Step

Because the full run is complete, the next step is to decide whether Qwen-LoRA should distill the Day9 pointwise relevance schema, the Day10 list schema, or both.
