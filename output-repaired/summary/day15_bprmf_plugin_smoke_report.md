# Day15 BPR-MF Personalized Backbone Plug-in Smoke Report

## 1. Day14 Recap

Day14 used a train-only item popularity backbone to unblock the external plug-in engineering path. That result should not be interpreted as Scheme 4 failing on external backbones, because popularity is not user-specific and its misranks are only weakly related to evidence risk.

## 2. Why BPR-MF

Day15 replaces popularity with BPR-MF. This is still lightweight, but it is a real personalized recommender with user and item embeddings:

`score(u, i) = dot(user_embedding[u], item_embedding[i])`

It is easier to train and export than SASRec, while being much closer to a recommendation backbone than global popularity.

## 3. Training Setup

Training data:

`data\processed\amazon_beauty\train.jsonl`

Only train split positive interactions are used as positives. Negative items are sampled from the train item universe excluding each user's train positives. No valid/test labels are used for training.

Hyperparameters:

- embedding_dim: `64`
- epochs: `20`
- batch_size: `1024`
- learning_rate: `0.001`
- seed: `42`
- final training loss: `0.652338`

Checkpoint path:

`artifacts/backbones/bprmf_beauty_100/bprmf.pt`

The checkpoint is an artifact and should not be committed.

## 4. Score Export

Score export:

`output-repaired/backbone/bprmf_beauty_100/candidate_scores.csv`

Rows: `600`

Users: `100`

Fallback score rows: `216`

Fields include:

`user_id, candidate_item_id, backbone_score, label, backbone_rank, split, backbone_name, raw_user_id, raw_item_id, mapped_user_id, mapped_item_id, mapping_success, fallback_score, fallback_reason`

## 5. Join Diagnostics

Joined table:

`output-repaired/summary/day15_bprmf_beauty_100_joined_candidates.csv`

Join coverage: `1.0000`

Missing evidence rows: `0`

## 6. Plug-in Result

BPR-MF only:

`{'method': 'A_BPRMF_only', 'backbone_name': 'bprmf', 'lambda': 0.0, 'alpha': 0.75, 'beta': 0.25, 'normalization': 'minmax', 'HR@10': 1.0, 'NDCG@10': 0.5248024144632589, 'MRR@10': 0.3756666666666667, 'Recall@10': 1.0, 'rank_change_rate': 0.0, 'top10_order_change_rate': 0.0, 'mean_kendall_tau': 0.9999999999999999, 'base_risk_spearman': -0.10851282927709349}`

Best grid row:

`{'method': 'D_BPRMF_plus_calibrated_relevance_plus_evidence_risk', 'backbone_name': 'bprmf', 'lambda': 0.5, 'alpha': 0.75, 'beta': 0.25, 'normalization': 'zscore', 'HR@10': 1.0, 'NDCG@10': 0.5934619966770437, 'MRR@10': 0.4660000000000001, 'Recall@10': 1.0, 'rank_change_rate': 0.6583333333333333, 'top10_order_change_rate': 0.97, 'mean_kendall_tau': 0.4640000000000001, 'base_risk_spearman': -0.10851282927709349}`

Diagnostics:

`{'backbone_score_AUROC': 0.51214, 'calibrated_relevance_AUROC': 0.60096, 'evidence_risk_AUROC_for_error_or_misrank': 0.50905, 'backbone_risk_spearman': -0.060062707758239686, 'risk_mean_for_wrong_high_rank': 0.43513333333739995, 'risk_mean_for_correct_high_rank': 0.4256666666699999, 'best_method': 'D_BPRMF_plus_calibrated_relevance_plus_evidence_risk', 'best_normalization': 'zscore', 'best_lambda': 0.5, 'best_NDCG@10': 0.5934619966770437, 'best_MRR@10': 0.4660000000000001, 'best_relative_NDCG_vs_backbone': 0.130829394685629, 'best_relative_MRR_vs_backbone': 0.24046140195208526, 'bprmf_backbone_NDCG@10': 0.5248024144632589, 'bprmf_backbone_MRR@10': 0.3756666666666667, 'day14_popularity_best_NDCG@10': 0.5541133161324351, 'day14_popularity_best_MRR@10': 0.4125, 'bprmf_vs_day14_popularity_NDCG_delta': -0.02931090166917616, 'bprmf_vs_day14_popularity_MRR_delta': -0.036833333333333274}`

## 7. Interpretation

The main checks are whether BPR-MF is stronger than Day14 popularity, whether calibrated relevance is complementary to BPR-MF, and whether evidence risk identifies wrong high-rank candidates. This is still a 100-user smoke test, not a full external SOTA result.

## 8. Day16 Recommendation

If BPR-MF improves over popularity and any Scheme-4 variant improves over BPR-MF-only, Day16 should scale to larger/full Beauty. If BPR-MF is weak or Scheme-4 does not help, Day16 should either tune BPR-MF lightly or move to SASRec as the stronger sequential backbone.
