# Day17 SASRec Backbone Plug-in Smoke Report

## 1. Why Stop BPR-MF Full

BPR-MF remained confounded by high fallback and weak backbone-only ranking. Day17 therefore uses a sequence-based backbone that scores candidates from history rather than user-id embeddings.

## 2. Why Minimal SASRec

The LLMEmb SASRec implementation exists but its official pipeline still needs missing handled data, embeddings, and checkpoints. A minimal SASRec-style encoder can be trained honestly from the current Beauty train split and can export candidate scores for the Day9 evidence pool.

## 3. Training And Score Export

Training uses only positive train rows from `data/processed/amazon_beauty/train.jsonl`. History titles are mapped through `items.csv`; test labels are not used for training. Candidate scores are exported to `output-repaired/backbone/sasrec_beauty_100/candidate_scores.csv`.

## 4. Join Coverage / Fallback

Join coverage: `1.0000`.

Fallback rate: `0.1067`.

Positive fallback rate: `0.1800`.

Negative fallback rate: `0.0920`.

Compared with earlier smoke backbones:

Day14 train-popularity NDCG@10 `0.5541`, MRR@10 `0.4125`. Day16 repaired BPR-MF-only NDCG@10 `0.5254`, MRR@10 `0.3763`, fallback `0.4583`.

The minimal SASRec backbone is healthier than BPR-MF mainly because fallback drops below 20%. Its standalone ranking is only comparable to train-popularity on this 100-user slice, so this remains a smoke validation rather than a final external performance claim.

## 5. Backbone Only Vs Scheme 4 Plug-in

Best plug-in diagnostics:

`{'backbone_score_AUROC': 0.51118, 'calibrated_relevance_AUROC': 0.60096, 'evidence_risk_AUROC_for_error_or_misrank': 0.50905, 'backbone_risk_spearman': 0.04759951932600549, 'fallback_rate': 0.10666666666666667, 'best_method': 'B_Backbone_plus_calibrated_relevance', 'best_normalization': 'minmax', 'best_lambda': 0.0, 'best_alpha': 0.5, 'best_beta': 0.5, 'best_NDCG@10': 0.592421108470827, 'best_MRR@10': 0.4626666666666667, 'best_HR@10': 1.0, 'best_relative_NDCG_vs_backbone': 0.0746342282308898, 'best_relative_MRR_vs_backbone': 0.1316754993885039}`

Best row per method:

| method | NDCG@10 | MRR@10 | lambda | alpha | beta | normalization | rank_change_rate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A_Backbone_only | 0.5513 | 0.4088 | 0.00 | 1.00 | 0.00 | minmax | 0.0000 |
| B_Backbone_plus_calibrated_relevance | 0.5924 | 0.4627 | 0.00 | 0.50 | 0.50 | minmax | 0.6600 |
| C_Backbone_plus_evidence_risk | 0.5513 | 0.4088 | 0.00 | 1.00 | 0.00 | minmax | 0.0000 |
| D_Backbone_plus_calibrated_relevance_plus_evidence_risk | 0.5924 | 0.4627 | 0.00 | 0.50 | 0.50 | minmax | 0.6600 |

The best row uses calibrated relevance without a positive evidence-risk penalty when `lambda = 0`. This again supports the current interpretation: Scheme 4's most reliable contribution is calibrated relevance posterior; evidence risk remains a secondary regularizer and does not yet provide standalone ranking gains in this slice.

## 6. Day18 Decision

Because fallback is below 20% and the plug-in path is technically clean, Day18 can expand to 500 users. The expansion should still be framed as larger smoke validation, not full SOTA comparison, because SASRec-only is not yet clearly stronger than train-popularity.
