# Day21 Second Backbone Plug-in Smoke Report

## 1. Why Second Backbone After Day20

Day20 showed that Scheme 4 works as a plug-in for the minimal SASRec-style backbone. Day21 checks whether the same plug-in path can attach to a second external/NH backbone implementation.

## 2. Selected Backbone And Why

Selected backbone: LLM-ESR GRU4Rec.

OpenP5 was audited first, but its useful candidate scoring path would require downloaded generated data/checkpoints and generative score adaptation. LLM-ESR's base GRU4Rec class exposes direct candidate logits through `predict()` and can be trained from the current Beauty train split without LLM embeddings.

## 3. Code Entrypoints

See `output-repaired/summary/day21_second_backbone_code_entrypoints.md`.

## 4. Score Export Schema

Candidate scores are exported to `output-repaired/backbone/second_backbone_beauty_100/candidate_scores.csv` with:

`user_id, candidate_item_id, backbone_score, label, backbone_rank, split, backbone_name, fallback_score, fallback_reason`.

## 5. Join / Fallback Diagnostics

Join coverage: `1.0000`.

Fallback rate: `0.1533`.

Positive fallback rate: `0.2100`.

Negative fallback rate: `0.1420`.

## 6. Backbone Only Vs Scheme 4 Plug-in

Best diagnostics:

`{'backbone_score_AUROC': 0.49224, 'calibrated_relevance_AUROC': 0.60096, 'evidence_risk_AUROC_for_error_or_misrank': 0.50905, 'backbone_risk_spearman': 0.019216554446559255, 'fallback_rate': 0.15333333333333332, 'best_method': 'D_Backbone_plus_calibrated_relevance_plus_evidence_risk', 'best_relative_NDCG_vs_backbone': 0.09010448647914729, 'best_relative_MRR_vs_backbone': 0.16425339366515837, 'best_lambda': 0.2, 'best_alpha': 0.5, 'best_beta': 0.5, 'best_normalization': 'minmax', 'best_NDCG@10': 0.5669772299935605, 'best_MRR@10': 0.42883333333333334, 'backbone_NDCG@10': 0.5201127387566313, 'backbone_MRR@10': 0.36833333333333335}`

Best row per method:

| method | NDCG@10 | MRR@10 | rel NDCG | rel MRR | lambda | alpha | beta | normalization | rank_change_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A_Backbone_only | 0.5201 | 0.3683 | 0.0000 | 0.0000 | 0.00 | 1.00 | 0.00 | minmax | 0.0000 |
| B_Backbone_plus_calibrated_relevance | 0.5575 | 0.4147 | 0.0719 | 0.1258 | 0.00 | 0.50 | 0.50 | minmax | 0.6667 |
| C_Backbone_plus_evidence_risk | 0.5340 | 0.3883 | 0.0267 | 0.0543 | 0.50 | 1.00 | 0.00 | zscore | 0.4983 |
| D_Backbone_plus_calibrated_relevance_plus_evidence_risk | 0.5670 | 0.4288 | 0.0901 | 0.1643 | 0.20 | 0.50 | 0.50 | minmax | 0.6550 |

## 7. Comparison With SASRec Result

This is a 100-user smoke on a second external implementation, not a full comparison. If join/fallback are healthy and gains are positive, Day22 can expand. If fallback is high or gains collapse, Day22 should diagnose history mapping and candidate cold-item behavior before scaling.

## 8. Day22 Recommendation

If the Day21 smoke is healthy, expand LLM-ESR GRU4Rec to 500 users with fixed settings. If it is unhealthy, switch to a different public sequential backbone with simpler data requirements.
