# Framework-Observation-Day1i Order-Shuffled Behavioral Uncertainty Report

## Why Day1i

Day1h was strong but confounded by positive-at-position-1 candidate order bias. Day1h cannot be used as clean evidence until this shuffle control is complete.

## Concept Boundary

Behavioral confidence / uncertainty is still confidence observation, but it is implicit rather than verbalized. It must be disentangled from candidate position bias before making stronger claims.

## Shuffle Setup

Day1i uses the same Beauty 100-user valid/test subset and the same six-candidate pools as Day1h. Candidate order is shuffled once per user with seed `42`. Evaluation uses `order_neutral_expected_tie_metric` and does not use input order for tie-breaking.

## Shuffled Ranking Result

- shuffled rank_score test MRR/HR@1/NDCG@3/AUROC: `0.49624999999999997` / `0.275` / `0.4730313019464342` / `0.58556`
- shuffled stability-weighted test MRR/HR@1/NDCG@3/AUROC: `0.465` / `0.26` / `0.3970208679642895` / `0.50958`

## Shuffled Behavioral Uncertainty

- majority_top1_confidence AUROC for top1 correctness: `0.6473616473616474`
- top1_vote_entropy AUROC for error risk: `0.6518661518661518`
- rank_variance inverse AUROC for correctness: `0.6083226083226083`

## Conclusion

`order_bias_confirmed`

If performance remains high, behavioral uncertainty is promising. If it drops near random, Day1h was mostly order bias. If ranking remains useful but uncertainty is weak, use listwise ranking score, not confidence.
