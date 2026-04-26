# Day39 Books/Electronics Small Replication Report

## 1. Why Books/Electronics Small

Day39 extends the small-domain continuity path from movies_small to books_small and electronics_small. These are cross-domain sanity experiments, not replacements for Beauty full or regular medium analysis.

## 2. API Fix

Day39 commands clear `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` before calling `main_infer.py`, inheriting the Day37 movies_small successful DeepSeek route. No prompt, parser, formula, backend, or model changes were made.

## 3. Calibration Result

| domain | raw ECE | calibrated ECE | raw Brier | calibrated Brier | raw AUROC | calibrated AUROC |
|---|---:|---:|---:|---:|---:|---:|
| books_small | 0.1477 | 0.0152 | 0.1380 | 0.1113 | 0.7577 | 0.7555 |
| electronics_small | 0.1690 | 0.0087 | 0.1705 | 0.1296 | 0.6461 | 0.6460 |

## 4. Backbone Plug-in Result

| domain | backbone | health | fallback | pos fallback | backbone NDCG | best method | best NDCG | best MRR | rel NDCG | rel MRR |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| books_small | sasrec | caution | 0.4440 | 0.9700 | 0.6352 | B_SASRec_plus_calibrated_relevance | 0.7447 | 0.6612 | 0.1723 | 0.2863 |
| books_small | gru4rec | fallback_heavy | 0.6363 | 0.9800 | 0.5394 | B_GRU4Rec_plus_calibrated_relevance | 0.6998 | 0.6032 | 0.2973 | 0.5463 |
| books_small | bert4rec | fallback_heavy | 0.6363 | 0.9800 | 0.5374 | B_Bert4Rec_plus_calibrated_relevance | 0.6834 | 0.5816 | 0.2717 | 0.5007 |
| electronics_small | sasrec | caution | 0.4813 | 0.9480 | 0.6497 | B_SASRec_plus_calibrated_relevance | 0.6818 | 0.5787 | 0.0495 | 0.0858 |
| electronics_small | gru4rec | fallback_heavy | 0.6707 | 0.9680 | 0.5859 | B_GRU4Rec_plus_calibrated_relevance | 0.6390 | 0.5231 | 0.0908 | 0.1606 |
| electronics_small | bert4rec | fallback_heavy | 0.6707 | 0.9680 | 0.5937 | B_Bert4Rec_plus_calibrated_relevance | 0.6380 | 0.5214 | 0.0746 | 0.1314 |

## 5. Fallback Health

`healthy` means fallback_rate < 0.2 and positive_fallback_rate < 0.2. `caution` and `fallback_heavy` rows should not be described as fully healthy ID-backbone evidence. They remain useful for directionality / compensation analysis.

## 6. Relation To Movies Small And Beauty

Beauty full three-backbone multi-seed remains the primary performance evidence. Small domains provide cross-domain sanity / continuity. If a small-domain backbone is fallback-heavy, interpret gains with the same caution introduced by Day38.

## 7. Limitations

Each small domain uses 6 candidates per user, so HR@10 is trivial and not used as a claim-supporting metric. Primary metrics are NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.

## 8. Day40 Recommendation

If books/electronics directions are consistent, Day40 should consolidate the small-domain cross-domain table and claim map. If any backbone is fallback-heavy, add fallback sensitivity before making stronger statements for that domain/backbone.
