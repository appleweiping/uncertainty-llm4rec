# Day7 Candidate Relevance Evidence Report

## 1. Motivation

The previous yes/no evidence posterior branch is a controlled confidence-repair benchmark. Day7 migrates the same evidence-posterior idea into a more recommendation-native candidate relevance scoring setting, where the primary model output is `relevance_probability` rather than a self-reported yes/no confidence.

## 2. Schema

`relevance_probability` estimates whether the candidate matches the user history. It is not named or treated as `raw_confidence`. Evidence fields remain explicit: `positive_evidence`, `negative_evidence`, `ambiguity`, and `missing_information`. The derived decision risk is `evidence_risk = (1 - abs_evidence_margin + ambiguity + missing_information) / 3`.

## 3. Parser And Inference

The Beauty DeepSeek relevance-evidence smoke test uses `beauty_deepseek_relevance_evidence_100` with 100 valid rows and 100 test rows. Valid parse_success is `1.0000` and test parse_success is `1.0000`. On the test split, average relevance_probability is `0.4447` and average evidence_risk is `0.4257`.

## 4. Calibration Smoke Test

On the test split, raw relevance has ECE `0.3007`, Brier `0.2692`, and AUROC `0.4826`. Valid-set calibrated relevance has ECE `0.0887`, Brier `0.1445`, and AUROC `0.5050`. Minimal evidence posterior relevance has ECE `0.1311`, Brier `0.1448`, and AUROC `0.4500`. Full evidence posterior relevance has ECE `0.1023`, Brier `0.1509`, and AUROC `0.4986`.

## 5. Rerank Smoke Test

The decoupled rerank smoke test runs lambda in {0, 0.1, 0.2} with `base_score = relevance_probability` and `uncertainty = evidence_risk`. The best row uses lambda `0` with NDCG@10 `0.5759`, MRR@10 `0.4431`, rank_change_rate `0.0000`, and top10_order_change_rate `0.0000`.

## 6. Limitation

This is a Beauty 100 schema smoke test, not a full-domain conclusion. Its purpose is to verify that candidate relevance scoring, evidence posterior calibration, and decoupled relevance-risk reranking can run end-to-end without confusing relevance probability with confidence.

## 7. Next Step

If the parser and calibration remain stable, Day8 should expand candidate relevance evidence to a larger Beauty sample or full Beauty. If parse_success or calibration is unstable, the next step should be prompt/parser repair before scaling.
