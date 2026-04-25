# Day16 BPR-MF Coverage Repair Report

## 1. Day15 Recap

Day15 produced a positive smoke signal: `BPR-MF + calibrated relevance + evidence risk` improved over BPR-MF-only on the 100-user slice. However, Day15 also showed that BPR-MF-only was weaker than Day14 popularity and had high fallback coverage. Therefore, Day16 focuses on coverage repair and component attribution rather than making a full performance claim.

## 2. Fallback Diagnosis

Day15 fallback rows: `275.0 / 600.0`.

Fallback rate: `0.4583`.

User fallback rows: `216.0`.

Item fallback rows: `84.0`.

Both user and item fallback rows: `25.0`.

Positive fallback rate: `0.5000`.

Negative fallback rate: `0.4500`.

The dominant issue is user coverage: some Day9 test users are not present in the BPR-MF train-positive mapping. Under a pure user-id BPR-MF model, these users cannot receive trained personalized embeddings without using non-train information. This is why BPR-MF-only can be underestimated.

## 3. Repair Strategy

Day16 keeps training leakage-safe: BPR-MF training still uses only train split positives. The repair expands vocabulary for query-time candidate users/items, but does not train on valid/test labels. Cold item strategies are explicit:

- `min_score`
- `train_popularity`
- `mean_embedding`

Unknown users remain fallback users because a user-id embedding model cannot personalize them without train history.

## 4. Repaired 100-user Result

Best cold strategy:

`{'cold_strategy': 'mean_embedding', 'backbone_NDCG@10': 0.5253896148533593, 'backbone_MRR@10': 0.3763333333333334, 'best_method': 'D_BPRMF_plus_calibrated_relevance_plus_evidence_risk', 'best_lambda': 0.2, 'best_alpha': 0.5, 'best_beta': 0.5, 'best_normalization': 'zscore', 'best_NDCG@10': 0.608550054877994, 'best_MRR@10': 0.4841666666666667, 'best_relative_NDCG_vs_backbone': 0.15828337232711653, 'best_relative_MRR_vs_backbone': 0.28653675819309105, 'fallback_rate': 0.4583333333333333, 'fallback_rate_positive': 0.5, 'fallback_rate_negative': 0.45}`

Selected repaired strategy output:

`output-repaired/backbone/bprmf_beauty_100_repaired/candidate_scores.csv`

Join coverage: `1.0000`

Fallback rate: `0.4583`

Best plug-in diagnostics:

`{'backbone_score_AUROC': 0.50736, 'calibrated_relevance_AUROC': 0.60096, 'evidence_risk_AUROC_for_error_or_misrank': 0.50905, 'backbone_risk_spearman': -0.0032911239954647375, 'fallback_rate': 0.4583333333333333, 'fallback_rate_positive': 0.5, 'fallback_rate_negative': 0.45, 'risk_mean_for_wrong_high_rank': 0.43513333333739995, 'risk_mean_for_correct_high_rank': 0.4256666666699999, 'best_method': 'D_BPRMF_plus_calibrated_relevance_plus_evidence_risk', 'best_normalization': 'zscore', 'best_lambda': 0.2, 'best_alpha': 0.5, 'best_beta': 0.5, 'best_NDCG@10': 0.608550054877994, 'best_MRR@10': 0.4841666666666667, 'best_HR@10': 1.0, 'best_relative_NDCG_vs_backbone': 0.15828337232711653, 'best_relative_MRR_vs_backbone': 0.28653675819309105, 'best_relative_HR_vs_backbone': 0.0, 'backbone_NDCG@10': 0.5253896148533593, 'backbone_MRR@10': 0.3763333333333334, 'backbone_HR@10': 1.0}`

Popularity comparison:

Day14 train-popularity backbone NDCG@10 was `0.5541` and MRR@10 was `0.4125`. The selected repaired BPR-MF-only NDCG@10 is `0.5254` and MRR@10 is `0.3763`, so this 100-user BPR-MF backbone is still not a healthy stronger baseline than popularity.

## 5. Larger 500-user Result

Ran 500-user smoke: `False`

If this is false, it is because repaired 100-user fallback remained too high for a clean larger interpretation.

## 6. Component Attribution

Day16 explicitly compares:

- B-only: calibrated relevance only
- C-only: evidence risk only
- D: calibrated relevance + evidence risk

Best row per method group:

| method | NDCG@10 | MRR@10 | lambda | alpha | beta | normalization | rank_change_rate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A_BPRMF_only | 0.5254 | 0.3763 | 0.00 | 1.00 | 0.00 | minmax | 0.0000 |
| B_BPRMF_plus_calibrated_relevance | 0.6074 | 0.4818 | 0.00 | 0.50 | 0.50 | minmax | 0.7350 |
| C_BPRMF_plus_evidence_risk | 0.5764 | 0.4432 | 0.05 | 0.50 | 0.50 | minmax | 0.3500 |
| D_BPRMF_plus_calibrated_relevance_plus_evidence_risk | 0.6086 | 0.4842 | 0.20 | 0.50 | 0.50 | zscore | 0.7350 |

B-only reaches NDCG@10 `0.6074`, C-only reaches `0.5764`, and D reaches `0.6086`. Most of the improvement comes from calibrated relevance; evidence risk is useful as a small regularizer when combined with calibrated relevance, but it is not a strong standalone error detector in this slice.

The clean interpretation is that Scheme 4 is currently strongest as calibrated relevance posterior first and evidence-risk regularization second. The risk signal alone remains close to random for this BPR-MF slice, partly because the backbone still has high user/item fallback.

## 7. Decision For Day17

If we want a stronger external backbone, the clean next step is SASRec because it can score from sequence history and should reduce unknown-user fallback. If we continue BPR-MF, Day17 should first repair user coverage or use a history-derived user representation; otherwise full-scale BPR-MF will remain confounded by cold users.
