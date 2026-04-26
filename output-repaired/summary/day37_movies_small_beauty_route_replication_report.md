# Day37 Movies Small Beauty-Route Replication Report

## 1. Why Movies Small

`movies_small` has a complete Beauty-compatible schema and healthy low cold-rate, so it is a clean small-domain replication of the Beauty Day9 -> backbone plug-in route. This is cross-domain sanity / continuity, not a replacement for regular medium or full-domain claims.

## 2. API Parity

Day37 reuses the same DeepSeek backend/config/command route as Day29/Day30. The earlier APIConnectionError was traced to bad proxy environment variables (`127.0.0.1:9`), not to prompt/schema/parser/config mismatch. Clearing proxy variables restored inference.

## 3. Relevance Calibration

On movies_small test, raw relevance ECE `0.3079` and Brier `0.2337` improved to calibrated ECE `0.0065` and Brier `0.1323`. AUROC changed from `0.5854` to `0.5946`. This matches the Beauty interpretation: calibration fixes probability quality more than ranking separability.

## 4. Three Backbone Plug-in

| backbone | fallback_rate | backbone NDCG@10 | backbone MRR | best method | best NDCG@10 | best MRR | rel NDCG | rel MRR |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| sasrec | 0.4100 | 0.5411 | 0.3925 | D_SASRec_plus_calibrated_relevance_plus_evidence_risk | 0.6143 | 0.4887 | 0.1353 | 0.2452 |
| gru4rec | 0.5710 | 0.5055 | 0.3487 | B_GRU4Rec_plus_calibrated_relevance | 0.5842 | 0.4502 | 0.1557 | 0.2911 |
| bert4rec | 0.5710 | 0.4862 | 0.3235 | D_Bert4Rec_plus_calibrated_relevance_plus_evidence_risk | 0.5750 | 0.4388 | 0.1825 | 0.3566 |

Important caveat: join coverage is 1.0, but ID-backbone fallback remains non-trivial, especially on positives. This means the movies_small replication supports cross-domain sanity and directionality, but it should not be presented as a fully healthy external-backbone benchmark without a later mapping/training-vocab repair.

## 5. Component Attribution

For each backbone, compare B/C/D rows in the grid. The expected CEP interpretation remains: calibrated relevance is the primary posterior signal; evidence_risk is a secondary regularizer if D improves over B at positive lambda. C-only should not be treated as the main scorer unless it beats B consistently.

## 6. Direction Against Beauty

Movies_small now follows the same route as Beauty: DeepSeek candidate relevance evidence, valid-fit/test-eval calibration, and three sequential backbone plug-in grids. This supports cross-domain continuity, but only at small-domain sanity scale.

## 7. Limitations

Movies_small has 6 candidates per user, so HR@10 is trivial and is not used as primary evidence. This result does not replace regular medium/full-domain analysis.

## 8. Next Step

If the three backbone directions are positive, Day38 can run movies_small multi-seed or extend the same small-domain sanity path to books_small/electronics_small one domain at a time.
