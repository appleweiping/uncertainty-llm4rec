# Day22 LLM-ESR GRU4Rec Full Report

## 1. Day21 Recap

Day21 selected LLM-ESR GRU4Rec as the second external/NH backbone smoke because it exposes real candidate logits through `GRU4Rec.predict()` without requiring OpenP5-style generative checkpoint adaptation.

## 2. Day22 Setup

Day22 expands the same LLM-ESR GRU4Rec adapter to the full Beauty evidence-aligned candidate pool. Training uses only Beauty train split. No DeepSeek API calls, no prompt changes, no LoRA, and no formula changes are used.

## 3. Backbone Health

Health status: `healthy`.

Full users: `973`.

Backbone rows: `5838`.

Join coverage: `1.0000`.

Fallback rate: `0.1336`.

Positive fallback rate: `0.1984`.

Negative fallback rate: `0.1207`.

GRU4Rec-only NDCG@10: `0.5329`.

GRU4Rec-only MRR@10: `0.3847`.

## 4. Full Plug-in Result

Best method: `D_Backbone_plus_calibrated_relevance_plus_evidence_risk`.

Best NDCG@10: `0.6020`.

Best MRR@10: `0.4756`.

Relative NDCG improvement vs GRU4Rec-only: `0.1297`.

Relative MRR improvement vs GRU4Rec-only: `0.2362`.

Best row per method:

| method | HR@10 | NDCG@10 | MRR@10 | rel NDCG | rel MRR | lambda | alpha | beta | normalization | rank_change_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A_Backbone_only | 1.0000 | 0.5329 | 0.3847 | 0.0000 | 0.0000 | 0.00 | 1.00 | 0.00 | minmax | 0.0000 |
| B_Backbone_plus_calibrated_relevance | 1.0000 | 0.5910 | 0.4609 | 0.1091 | 0.1980 | 0.00 | 0.50 | 0.50 | zscore | 0.6238 |
| C_Backbone_plus_evidence_risk | 1.0000 | 0.5437 | 0.3997 | 0.0203 | 0.0389 | 0.50 | 1.00 | 0.00 | zscore | 0.4889 |
| D_Backbone_plus_calibrated_relevance_plus_evidence_risk | 1.0000 | 0.6020 | 0.4756 | 0.1297 | 0.2362 | 0.20 | 0.50 | 0.50 | zscore | 0.6394 |

## 5. Component Attribution

If B is close to D, calibrated relevance posterior is the primary contributor. If D exceeds B at a positive lambda, evidence risk contributes as secondary regularization.

## 6. Comparison With SASRec Full Multi-seed

Day20 best SASRec multi-seed method was `D_SASRec_plus_calibrated_relevance_plus_evidence_risk` with mean NDCG@10 `0.6099` and mean MRR@10 `0.4853`. Day22 checks the same plug-in direction on LLM-ESR GRU4Rec; exact numeric equality is not expected.

## 7. Claim Boundary

This is second external sequential backbone full validation, not an external SOTA leaderboard claim.

## 8. Day23 Recommendation

If Day22 is healthy and positive, Day23 should produce a two-backbone final claim map and main table. If Day22 is partially blocked, Day23 should document the limitation and decide whether a third public backbone is needed.

Case study counts: `{'promoted_positive': 10, 'demoted_negative': 10, 'corrected_positive': 10, 'harmed_positive': 10, 'high_risk_demoted': 10}`.
