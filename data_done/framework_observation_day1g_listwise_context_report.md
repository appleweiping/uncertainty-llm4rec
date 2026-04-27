# Framework-Observation-Day1g Relative Context Audit Report

## Scope

Day1g tests whether relative candidate context improves local Qwen-LoRA confidence/relevance signals. It uses Beauty only, no training, no evidence fields, no CEP, no external API, and no four-domain run.

## Prior Observations

- Pointwise verbalized confidence collapsed.
- Pointwise logit P(true) is usable but weak.
- Self-consistency is not the primary confidence line. After tie-aware ranking fix, self-consistency no longer beats logit P(true). It is more expensive and weaker than logit on the same subset.

## Test Comparison

- pointwise logit P(true): MRR `0.48083333333333333`, HR@1 `0.26`, NDCG@3 `0.4422580581071478`
- listwise ranking: MRR `0.73`, HR@1 `0.46`, NDCG@3 `0.800702066928587`
- pairwise win rate: MRR `0.4653333333333333`, HR@1 `0.21`, NDCG@3 `0.4648766531785769`

## Observation Finding

Relative candidate context is useful for local Qwen-LoRA recommendation ranking, but raw self-reported uncertainty is still unusable. The observation claim is: use relative ranking score, but not raw verbalized uncertainty.

This is not a CEP success claim and not a final method claim.

## Recommendation

`use_relative_ranking_score_but_not_raw_uncertainty`
