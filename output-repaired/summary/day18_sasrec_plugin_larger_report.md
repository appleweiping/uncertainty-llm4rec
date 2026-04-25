# Day18 SASRec Plug-in Larger Validation Report

## 1. Day17 Recap

Day17 established a healthier 100-user sequential backbone smoke: join coverage was 1.0, fallback dropped below 20%, and adding calibrated relevance improved SASRec-style ranking.

## 2. Day18 Setup

Day18 expands the same minimal SASRec-style backbone to 500 Beauty users. Training still uses only the train split, and this is still a larger smoke validation rather than a full SOTA comparison.

## 3. Backbone Health

Join coverage: `1.0000`.

Fallback rate: `0.0883`.

Positive fallback rate: `0.1740`.

Negative fallback rate: `0.0712`.

SASRec-only NDCG@10: `0.5491`.

SASRec-only MRR@10: `0.4057`.

## 4. Plug-in Result

Best method: `D_SASRec_plus_calibrated_relevance_plus_evidence_risk`.

Best NDCG@10: `0.6201`.

Best MRR@10: `0.4988`.

Relative NDCG improvement vs SASRec-only: `0.1293`.

Relative MRR improvement vs SASRec-only: `0.2295`.

Best row per method:

| method | NDCG@10 | MRR@10 | rel NDCG | rel MRR | lambda | alpha | beta | normalization | rank_change_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A_SASRec_only | 0.5491 | 0.4057 | 0.0000 | 0.0000 | 0.00 | 1.00 | 0.00 | minmax | 0.0000 |
| B_SASRec_plus_calibrated_relevance | 0.6139 | 0.4903 | 0.1181 | 0.2084 | 0.00 | 0.50 | 0.50 | minmax | 0.6510 |
| C_SASRec_plus_evidence_risk | 0.5620 | 0.4227 | 0.0235 | 0.0419 | 0.50 | 1.00 | 0.00 | minmax | 0.4913 |
| D_SASRec_plus_calibrated_relevance_plus_evidence_risk | 0.6201 | 0.4988 | 0.1293 | 0.2295 | 0.20 | 0.50 | 0.50 | zscore | 0.6673 |

## 5. Component Attribution

If the best B row remains close to or better than D, the gain is mainly from calibrated relevance posterior. If D improves over B at positive lambda, evidence risk is adding regularization. The diagnostics file records evidence-risk AUROC and base-risk Spearman for this check.

## 6. Case Study

Case study rows were written to `output-repaired/summary/day18_sasrec_plugin_case_study.csv`.

Case type counts: `{'promoted': 10, 'demoted': 10, 'corrected_positive': 10, 'harmed_positive': 10}`.

## 7. Decision For Day19

If join coverage stays above 0.95, fallback remains below 20%, and relative gains remain positive, Day19 can either expand to full or run multi-seed/slice stability. If evidence risk still contributes little beyond calibrated relevance, the paper should frame Scheme 4 as calibrated relevance posterior first and evidence-risk regularization second.
