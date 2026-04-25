# Day5 Repair Report: Evidence Rerank Diagnosis and Decoupled Decision Experiment

This Day5 repair freezes the rerank finding as a mechanism-level result rather than treating the first lambda grid as a failed evidence posterior experiment. The original monotonic rerank used `minimal_repaired_confidence` as the base score and `minimal_evidence_uncertainty = 1 - minimal_repaired_confidence` as the uncertainty penalty. Therefore:

```text
final_score = repaired_confidence - lambda * (1 - repaired_confidence)
            = (1 + lambda) * repaired_confidence - lambda
```

For every non-negative lambda, this is a monotonic affine transform of `repaired_confidence`. It cannot change within-user ranking order. The diagnosis confirms this exactly: `rank_change_rate = 0`, `top10_change_rate = 0`, `top10_order_change_rate = 0`, `mean_kendall_tau = 1`, and `base_uncertainty_spearman = -1`. This is not evidence that evidence posterior is useless. It shows that the original formulation tied relevance score and uncertainty into opposite sides of the same scalar, making the uncertainty penalty mathematically unable to produce a decision change.

The repaired Day5 version therefore introduces a decoupled rerank branch. Instead of deriving the base score and penalty from the same scalar, it evaluates relevance-like scores and risk-like uncertainty signals separately. The strongest current setting is setting B:

```text
base_score = raw_confidence
uncertainty = evidence_risk
evidence_risk = (1 - abs_evidence_margin + ambiguity + missing_information) / 3
normalization = per-user minmax
lambda = 0.2
```

Under this decoupled setting, NDCG@10 improves from `0.4993` to `0.5944`, a relative improvement of `19.04%` over the current full Beauty internal monotonic baseline. MRR@10 improves from `0.3428` to `0.4649`, a relative improvement of `35.60%`. The rerank now actually changes decisions, with `rank_change_rate = 0.3390`. These gains are internal to the full Beauty evidence-posterior rerank comparison; they are not claims against external SOTA.

The Day6-Day10 path should continue from this repaired formulation. Day6 should aggregate the monotonic no-op diagnosis and the decoupled grid into the official ablation table. Day7 should perform case studies for setting B, especially user-candidate pairs that are strongly promoted or demoted, checking whether evidence risk is behaviorally sensible. Day8 should run robustness and sensitivity across `minmax` vs `zscore`, lambda values `0.1/0.2/0.5`, and settings A/B/C. Day9 should prepare the migration path toward candidate relevance scoring or a Qwen-LoRA evidence generator. Day10 should summarize the phase with the precise claim: evidence posterior repairs uncertainty quality, and decoupled relevance-risk reranking converts that repaired uncertainty into decision payoff.
