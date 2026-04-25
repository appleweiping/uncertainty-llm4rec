# Day8 Relevance Evidence Medium Report

## 1. Day7 Recap

Day7 verified that the candidate relevance evidence schema, parser, valid-set calibration, and decoupled rerank smoke test can run on Beauty 100. Day8 scales this path to a medium Beauty sample without moving to full data, LoRA, or external baselines.

## 2. Day8 Setup

The experiment `beauty_deepseek_relevance_evidence_1000` uses `prompts/candidate_relevance_evidence.txt` with 1000 valid rows and 1000 test rows. The main score remains `relevance_probability`, not `raw_confidence`. Calibration produces `calibrated_relevance_probability`, and uncertainty is represented by `relevance_uncertainty` or the decoupled `evidence_risk` used for reranking.

## 3. Parse And Field Diagnostics

Valid parse_success is `1.0000` and test parse_success is `1.0000`. On the test split, relevance_probability has mean `0.4250` and std `0.2178`, while evidence_risk has mean `0.4313` and std `0.1247`. The field diagnostics table records mean/std/min/max/quantiles and near-extreme rates for every evidence field.

## 4. Calibration Result

Raw relevance_probability has ECE `0.2580`, Brier `0.2269`, and AUROC `0.6147`. Valid-set calibrated relevance has ECE `0.0286`, Brier `0.1335`, and AUROC `0.6169`. Minimal evidence posterior has ECE `0.0296`, Brier `0.1355`, and AUROC `0.6104`. Full evidence posterior has ECE `0.0292`, Brier `0.1339`, and AUROC `0.6192`.

## 5. Rerank Result

The best medium-sample rerank row is setting `R-C` with `minmax` normalization and lambda `0.1`. It reaches NDCG@10 `0.6416`, MRR@10 `0.5272`, rank_change_rate `0.4040`, and top10_order_change_rate `0.7784`. Day8 does not compare directly against Day6 because Day6 is the yes/no full-Beauty branch and Day8 is the relevance medium branch.

## 6. Case Study

The case study exports promoted, demoted, and high-risk-demoted rows for the best nonzero-change setting. In the high-risk-demoted group, average rank_delta is `-2.1000`; negative values indicate that high-risk candidates move down under the decoupled evidence-risk penalty.

## 7. Decision For Day9

If the calibration rows show stable ECE/Brier improvement and rerank rows show nonzero rank changes, Day9 can expand relevance evidence to full Beauty or prepare the Qwen-LoRA relevance generator. If calibration is unstable or fields collapse toward extremes, the next step should be prompt/schema/feature repair before full scaling.
