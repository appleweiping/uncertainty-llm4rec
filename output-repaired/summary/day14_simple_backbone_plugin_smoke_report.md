# Day14 Simple External Backbone Plug-in Smoke Report

## 1. Why LLMEmb Is Temporarily Blocked

Day13 cloned and audited LLMEmb, and located the true score path in `external/LLMEmb/trainers/sequence_trainer.py`. However, the official clone does not include Beauty handled data, LLM item embeddings, SRS item embeddings, or a trained checkpoint. Therefore, LLMEmb remains a stronger external baseline target, but it should not block the first plug-in smoke test.

## 2. Why Simple Backbone First

Day14 uses a train-only item popularity backbone named `train_popularity`. It is not claimed as a strong SOTA baseline. Its role is to produce a real, reproducible `backbone_score` that is independent of Day9/Day10 scores and can be joined with Scheme-4 evidence fields.

## 3. Backbone Construction And Leakage Control

The backbone uses only positive rows from:

`data/processed/amazon_beauty/train.jsonl`

For each item:

`backbone_score = log(1 + train_positive_count)`

No valid/test labels are used to construct the score. Candidates unseen in train receive `backbone_score = 0` and `fallback_score = 1`.

## 4. Score Export Schema

Score export:

`output-repaired/backbone/simple_beauty_100/candidate_scores.csv`

Rows: `600`

Users: `100`

Fields include:

`user_id, candidate_item_id, backbone_score, label, backbone_rank, split, backbone_name, train_positive_count, fallback_score`

## 5. Day9 Evidence Join

Joined table:

`output-repaired/summary/day14_simple_backbone_beauty_100_joined_candidates.csv`

Join coverage: `1.0000`

Missing evidence rows: `0`

Fallback score rows: `84`

Because this backbone reuses Day9 candidate rows, candidate alignment is clean enough for smoke-test interpretation.

## 6. Plug-in Rerank Smoke Result

Backbone-only best row:

`{'method': 'A_Backbone_only', 'backbone_name': 'train_popularity', 'lambda': 0.0, 'alpha': 0.75, 'beta': 0.25, 'normalization': 'minmax', 'HR@10': 1.0, 'NDCG@10': 0.5541133161324351, 'MRR@10': 0.4125, 'Recall@10': 1.0, 'rank_change_rate': 0.0, 'top10_order_change_rate': 0.0, 'mean_kendall_tau': 0.9999999999999999, 'base_risk_spearman': 0.03220043573045305}`

Best plug-in/grid row:

`{'method': 'A_Backbone_only', 'backbone_name': 'train_popularity', 'lambda': 0.0, 'alpha': 0.75, 'beta': 0.25, 'normalization': 'minmax', 'HR@10': 1.0, 'NDCG@10': 0.5541133161324351, 'MRR@10': 0.4125, 'Recall@10': 1.0, 'rank_change_rate': 0.0, 'top10_order_change_rate': 0.0, 'mean_kendall_tau': 0.9999999999999999, 'base_risk_spearman': 0.03220043573045305}`

Diagnostics:

`{'backbone_score_AUROC': 0.42014, 'calibrated_relevance_AUROC': 0.60096, 'evidence_risk_AUROC_for_error_or_misrank': 0.50905, 'backbone_risk_spearman': 0.11741670801389478, 'risk_mean_for_wrong_high_rank': 0.43513333333739995, 'risk_mean_for_correct_high_rank': 0.4256666666699999, 'best_method': 'A_Backbone_only', 'best_normalization': 'minmax', 'best_lambda': 0.0, 'best_NDCG@10': 0.5541133161324351, 'best_MRR@10': 0.4125, 'best_relative_NDCG_vs_backbone': 0.0, 'best_relative_MRR_vs_backbone': 0.0}`

The result should be read as a technical plug-in smoke test, not an external SOTA claim. If Scheme-4 improves over this weak backbone, it shows the adapter can exploit evidence fields. If not, the pipeline is still useful because score export, evidence join, and grid evaluation are now unblocked.

## 7. Day15 Recommendation

If the goal is quick scale-up, run the same simple backbone on full Beauty candidate pools to verify stability. If the goal is stronger evidence, implement BPR-MF or SASRec next. LLMEmb should resume only after its handled Beauty data, embeddings, checkpoint, and reversible id mapping are prepared.
