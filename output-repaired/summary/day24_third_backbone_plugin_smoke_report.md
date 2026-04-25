# Day24 Third Backbone Plug-in Smoke Report

## 1. Why A Third Backbone After Day20/Day23

Day20 and Day23 established full multi-seed support on SASRec-style and LLM-ESR GRU4Rec. Day24 adds a third 100-user smoke to check that the plug-in path is not only a SASRec/GRU pattern.

## 2. Third Backbone Choice

Selected backbone: **LLM-ESR Bert4Rec**.

It is an external masked-transformer sequential recommender. It is not the current minimal SASRec-style backbone and not the LLM-ESR GRU4Rec model. It exposes real candidate logits via `Bert4Rec.predict()` and can train from the Beauty train split without missing checkpoints, LLM embeddings, or generated data.

## 3. Score Export And Join Diagnostics

Candidate scores: `output-repaired/backbone/third_backbone_beauty_100/candidate_scores.csv`.

Join coverage: `1.0000`.

Fallback rate: `0.1533`.

Positive fallback rate: `0.2100`.

Negative fallback rate: `0.1420`.

## 4. Backbone-only Vs B/C/D

Best row per method:

| method | NDCG@10 | MRR@10 | HR@10 | lambda | alpha | beta | normalization | rank_change_rate | relative_NDCG_vs_backbone | relative_MRR_vs_backbone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_Backbone_only | 0.5198 | 0.3682 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | minmax | 0.0000 | 0.0000 | 0.0000 |
| B_Backbone_plus_calibrated_relevance | 0.5927 | 0.4625 | 1.0000 | 0.0000 | 0.5000 | 0.5000 | minmax | 0.6667 | 0.1402 | 0.2562 |
| C_Backbone_plus_evidence_risk | 0.5393 | 0.3932 | 1.0000 | 0.2000 | 1.0000 | 0.0000 | minmax | 0.2567 | 0.0374 | 0.0679 |
| D_Backbone_plus_calibrated_relevance_plus_evidence_risk | 0.5950 | 0.4663 | 1.0000 | 0.1000 | 0.5000 | 0.5000 | minmax | 0.6667 | 0.1445 | 0.2666 |

Diagnostics:

`{'backbone_score_AUROC': 0.48898, 'calibrated_relevance_AUROC': 0.60096, 'evidence_risk_AUROC_for_error_or_misrank': 0.50905, 'backbone_risk_spearman': 0.08868777691373587, 'fallback_rate': 0.15333333333333332, 'fallback_rate_positive': 0.21, 'fallback_rate_negative': 0.142, 'best_method': 'D_Backbone_plus_calibrated_relevance_plus_evidence_risk', 'best_relative_NDCG_vs_backbone': 0.14450390542464658, 'best_relative_MRR_vs_backbone': 0.26663648709823456, 'best_lambda': 0.1, 'best_alpha': 0.5, 'best_beta': 0.5, 'best_normalization': 'minmax', 'best_NDCG@10': 0.5949631574328691, 'best_MRR@10': 0.4663333333333333, 'backbone_NDCG@10': 0.5198437109850833, 'backbone_MRR@10': 0.36816666666666664}`

## 5. Component Pattern

This is a 100-user smoke, not a full result. The main question is whether calibrated relevance remains the dominant useful Scheme 4 signal and whether evidence risk provides secondary regularization when combined with it.

## 6. Day25 Decision

If join coverage is at least 0.95, fallback is below 20%, and B/D improve over the backbone-only row, Day25 can expand Bert4Rec to 500/full. If Bert4Rec is unhealthy, the SASRec and GRU4Rec full multi-seed conclusions remain the main evidence, and Day25 should switch to a graph/MF-style public backbone.
