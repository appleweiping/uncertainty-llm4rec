# Day6 Decoupled Rerank Analysis

## 1. Day5-Repair Recap

The original monotonic rerank is a mathematical no-op, not a negative result for evidence posterior. It uses `repaired_confidence` as the base score and `1 - repaired_confidence` as the uncertainty penalty, so `final_score = repaired_confidence - lambda * (1 - repaired_confidence) = (1 + lambda) * repaired_confidence - lambda`. For non-negative lambda this preserves ranking order. The diagnosis records rank_change_rate = 0, top10_change_rate = 0, top10_order_change_rate = 0, mean_kendall_tau = 1, and base_uncertainty_spearman = -1.

## 2. Day6 Ablation

The monotonic internal baseline has NDCG@10 = `0.4993` and MRR@10 = `0.3428`. The best decoupled setting is setting `B` with `zscore` normalization and lambda `0.1`. It reaches NDCG@10 = `0.5952` and MRR@10 = `0.4659`, corresponding to internal relative improvements of `19.20%` in NDCG@10 and `35.89%` in MRR@10.

## 3. Sensitivity

Setting B is not a single-point accident in the current grid. Its best row uses lambda `0.1`, while the mean NDCG@10 across the tested lambdas for its best normalization is `0.5911` with std `0.0055`. This indicates that evidence_risk is useful across a local lambda region, although broader validation is still needed before making a general claim.

## 4. Mechanism

The best setting changes ranking order with rank_change_rate = `0.3248`, top10_order_change_rate = `0.6516`, and mean_kendall_tau = `0.8050`. Its top10_change_rate can stay low in this candidate-ranking setup because each user has at most ten candidates, so the more informative signal is top-10 order change. Setting B decouples raw confidence from evidence risk, where evidence_risk = (1 - abs_evidence_margin + ambiguity + missing_information) / 3. The case study includes promoted candidates, demoted candidates, and monotonic-high candidates demoted under high evidence risk. In the monotonic-high-risk-demoted group, average evidence_risk is `0.6000` and average rank_delta is `-3.4167`, which supports the intended mechanism: candidates with weak margins, ambiguity, or missing information are penalized.

## 5. Limitation

The reported improvement is an internal comparison against the current full Beauty monotonic baseline, not an external SOTA claim. The experiment still uses a yes/no pointwise-to-ranking formulation, and it has not yet been moved to candidate relevance scoring, local Qwen-LoRA evidence generation, or external recommendation baselines.

## 6. Next Step Recommendation

Day6 is stable enough to proceed to Day7 candidate relevance scoring prompt/schema design. The current evidence supports two claims: evidence posterior repairs uncertainty quality, and decoupled relevance-risk reranking can convert that repaired uncertainty into decision payoff. Day7 should not add external baselines yet; it should first migrate the signal formulation away from yes/no into candidate relevance scoring while preserving the evidence-risk mechanism.
