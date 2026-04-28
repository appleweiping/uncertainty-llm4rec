# Framework-Observation-Day1h Listwise Behavioral Uncertainty Report

## Scope

This is still Framework Observation. Day1h does not train, use evidence, implement CEP, call external APIs, or run four domains.

## Motivation

Relative listwise context improves recommendation ranking, while raw verbalized uncertainty collapses. Day1h tests whether behavioral uncertainty from repeated listwise rankings can replace self-reported confidence.

## Test Ranking Snapshot

- pointwise logit P(true) MRR/HR@1/NDCG@3: `0.48083333333333333` / `0.26` / `0.4422580581071478`
- Day1g single listwise MRR/HR@1/NDCG@3: `0.73` / `0.46` / `0.800702066928587`
- Day1h behavioral rank score MRR/HR@1/NDCG@3: `0.9075` / `0.815` / `0.9317220044107196`

## Behavioral Uncertainty

- majority_top1_confidence AUROC for top1 correctness: `0.7085385878489326`
- top1 vote entropy AUROC for error risk: `0.7196223316912972`
- rank entropy inverse AUROC for correctness: `0.5921592775041051`
- rank variance inverse AUROC for correctness: `0.7313218390804598`

## Recommendation

`use_rank_stability_as_behavioral_uncertainty_candidate`
