# Day20 SASRec Multi-seed Final Report

## 1. Day19 Recap

Day19 completed full Beauty SASRec plug-in validation with healthy join coverage and a stable positive gain from Scheme 4 plug-in scoring.

## 2. Multi-seed Setup

Seeds: `42`, `43`, `44`.

The comparison uses fixed Day19 settings rather than reselecting a best setting per seed:

- A: SASRec-only.
- B: minmax, alpha=0.5, beta=0.5, lambda=0.0.
- C: zscore, evidence-risk lambda=0.5.
- D: zscore, alpha=0.5, beta=0.5, lambda=0.2.

No DeepSeek API calls, prompt changes, LoRA, or formula tuning are used.

## 3. Stability Result

Mean fallback rate across seeds: `0.0884`.

| method | NDCG mean | NDCG std | MRR mean | MRR std | rel NDCG mean | rel MRR mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_SASRec_only | 0.5446 | 0.0028 | 0.3996 | 0.0038 | 0.0000 | 0.0000 |
| B_SASRec_plus_calibrated_relevance | 0.6053 | 0.0016 | 0.4790 | 0.0023 | 0.1114 | 0.1988 |
| C_SASRec_plus_evidence_risk | 0.5560 | 0.0048 | 0.4146 | 0.0065 | 0.0210 | 0.0378 |
| D_SASRec_plus_calibrated_relevance_plus_evidence_risk | 0.6099 | 0.0024 | 0.4853 | 0.0033 | 0.1199 | 0.2147 |

Best mean method: `D_SASRec_plus_calibrated_relevance_plus_evidence_risk`.

Best mean relative NDCG improvement: `0.1199`.

Best mean relative MRR improvement: `0.2147`.

## 4. Component Attribution

The component attribution table is written to `output-repaired/summary/day20_sasrec_component_attribution.csv`.

The expected interpretation is: calibrated relevance posterior is the primary contributor; evidence risk is a secondary regularizer when D improves over B, and a weak standalone scorer when C remains much smaller than B.

## 5. Final Claim

Scheme 4 can act as an external sequential backbone plug-in on full Beauty, primarily through calibrated relevance posterior, with evidence risk as secondary regularizer. This is not an external SOTA claim; it is a controlled full-domain plug-in validation.

## 6. Limitation

The backbone is a minimal SASRec-style implementation, not every NH/SOTA recommender. The next step is a second external ranking backbone or repository-level integration.

## 7. Day21 Recommendation

Day21 should select a second external backbone repository, preferably a healthier public sequential recommender that can export candidate scores without missing checkpoint or embedding dependencies.
