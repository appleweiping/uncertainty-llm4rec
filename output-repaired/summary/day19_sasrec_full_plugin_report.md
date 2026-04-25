# Day19 SASRec Full Beauty Plug-in Report

## 1. Day18 Recap

Day18 expanded the SASRec-style sequential backbone to 500 users and found stable positive plug-in gains with healthy join coverage and fallback.

## 2. Day19 Setup

Day19 runs the same SASRec-style backbone on the full Beauty candidate pool available in Day9 relevance evidence. Training still uses only the Beauty train split. No DeepSeek API calls, no LoRA, and no prompt changes are used.

## 3. Backbone Health

Full users: `973`.

Backbone rows: `5838`.

Join coverage: `1.0000`.

Fallback rate: `0.0884`.

Positive fallback rate: `0.1768`.

Negative fallback rate: `0.0707`.

SASRec-only HR@10: `1.0000`.

SASRec-only NDCG@10: `0.5415`.

SASRec-only MRR@10: `0.3954`.

## 4. Full Plug-in Result

Best method: `D_SASRec_plus_calibrated_relevance_plus_evidence_risk`.

Best HR@10: `1.0000`.

Best NDCG@10: `0.6122`.

Best MRR@10: `0.4886`.

Relative NDCG improvement vs SASRec-only: `0.1305`.

Relative MRR improvement vs SASRec-only: `0.2357`.

Best row per method:

| method | HR@10 | NDCG@10 | MRR@10 | rel NDCG | rel MRR | lambda | alpha | beta | normalization | rank_change_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A_SASRec_only | 1.0000 | 0.5415 | 0.3954 | 0.0000 | 0.0000 | 0.00 | 1.00 | 0.00 | minmax | 0.0000 |
| B_SASRec_plus_calibrated_relevance | 1.0000 | 0.6071 | 0.4816 | 0.1212 | 0.2181 | 0.00 | 0.50 | 0.50 | minmax | 0.6531 |
| C_SASRec_plus_evidence_risk | 1.0000 | 0.5602 | 0.4203 | 0.0345 | 0.0631 | 0.50 | 1.00 | 0.00 | zscore | 0.4827 |
| D_SASRec_plus_calibrated_relevance_plus_evidence_risk | 1.0000 | 0.6122 | 0.4886 | 0.1305 | 0.2357 | 0.20 | 0.50 | 0.50 | zscore | 0.6518 |

## 5. Component Attribution

If B remains close to D, calibrated relevance posterior is the primary contribution. If D exceeds B at positive lambda, evidence risk provides additional regularization. This report keeps that distinction explicit rather than treating Scheme 4 as one monolithic formula.

## 6. Case Study

Case study rows were written to `output-repaired/summary/day19_sasrec_full_plugin_case_study.csv`.

Case type counts: `{'promoted_positive': 10, 'demoted_negative': 10, 'corrected_positive': 10, 'harmed_positive': 10, 'high_risk_demoted': 10}`.

## 7. Claim Boundary

This is full Beauty external sequential backbone plug-in validation. It is not an external SOTA claim. The purpose is to test whether Scheme 4 can improve a healthy sequential backbone using existing Day9 evidence without additional API calls.

## 8. Day20 Recommendation

If the full result remains positive and diagnostics are healthy, Day20 should produce multi-seed or final-table stability, then prepare the final claim map. If full gains collapse, Day20 should do slice stability before any paper-level claim.
