# Day23 GRU4Rec Multi-seed And Two-backbone Report

## 1. Day22 Recap

Day22 showed that LLM-ESR GRU4Rec full single-seed validation was healthy and positive.

## 2. Multi-seed Setup

Seeds: `42`, `43`, `44`.

The settings are fixed from Day22 rather than reselected per seed:

- A: GRU4Rec-only.
- B: zscore, alpha=0.5, beta=0.5, lambda=0.
- C: zscore, lambda=0.5.
- D: zscore, alpha=0.5, beta=0.5, lambda=0.2.

No DeepSeek API calls, prompt changes, LoRA, or formula changes are used.

## 3. GRU4Rec Stability Result

| method | NDCG mean | NDCG std | MRR mean | MRR std | rel NDCG mean | rel MRR mean | fallback mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_GRU4Rec_only | 0.5347 | 0.0066 | 0.3873 | 0.0088 | 0.0000 | 0.0000 | 0.1336 |
| B_GRU4Rec_plus_calibrated_relevance | 0.5920 | 0.0036 | 0.4622 | 0.0049 | 0.1072 | 0.1938 | 0.1336 |
| C_GRU4Rec_plus_evidence_risk | 0.5454 | 0.0033 | 0.4018 | 0.0044 | 0.0201 | 0.0378 | 0.1336 |
| D_GRU4Rec_plus_calibrated_relevance_plus_evidence_risk | 0.6037 | 0.0025 | 0.4778 | 0.0032 | 0.1292 | 0.2342 | 0.1336 |

Best mean method: `D_GRU4Rec_plus_calibrated_relevance_plus_evidence_risk`.

Best mean relative NDCG improvement: `0.1292`.

Best mean relative MRR improvement: `0.2342`.

## 4. Two-backbone Comparison

The merged table is written to `output-repaired/summary/day23_two_backbone_external_plugin_main_table.csv`.

Both SASRec-style and LLM-ESR GRU4Rec support the same qualitative result: calibrated relevance posterior is the primary gain source, and evidence risk is a secondary regularizer.

## 5. Component Attribution

The attribution table is written to `output-repaired/summary/day23_gru4rec_component_attribution.csv`.

## 6. Day24 Recommendation

Day24 should audit a third backbone that is not just another GRU/SASRec clone. A graph or matrix-factorization-plus-neural baseline with clean candidate-score export would be ideal; ItemKNN/co-occurrence should only be a sanity fallback.
