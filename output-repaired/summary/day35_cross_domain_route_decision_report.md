# Day35 Cross-Domain Route Decision Report

## 1. Movies Cold-Rate Recap

Current Movies medium_5neg is cold-style sampling. Negative cold was driven by all-items negative sampling, while positive cold is also high due to chronological/domain long-tail effects.

## 2. Movies Content Carrier Result

Day34 showed TF-IDF/BM25 can score cold candidates without item-id embeddings. This is a cold-aware diagnostic carrier, not a SOTA backbone claim.

- bm25: A NDCG `0.6465`, B `0.6872`, C `0.6603`, D `0.6891`; contributor `calibrated_relevance_primary_evidence_risk_secondary`.
- tfidf_cosine: A NDCG `0.7075`, B `0.7372`, C `0.7158`, D `0.7378`; contributor `calibrated_relevance_primary_evidence_risk_secondary`.

## 3. Movies Warm Split Limitation

Warm negative sampling reduced negative cold to zero, but positive cold remained high. Warm-strict feasibility checks whether enough users have both valid/test positives inside train_backbone_vocab.

Movies warm-strict max users: `365`. Recommended users: `0`.

## 4. Books/Electronics Cold-Rate Audit

See `day35_books_electronics_medium_5neg_cold_rate_diagnostics.csv` for full details. The route summary is:

- movies: route `content_carrier_cold`, max warm-strict users `365`.
- books: route `content_carrier_cold`, max warm-strict users `54`.
- electronics: route `content_carrier_cold`, max warm-strict users `122`.

## 5. Recommended Day36 Route

If a domain has at least 1000 warm-strict users, use that warm-strict split for ID-based backbone + CEP. If not, use the cold-style content carrier route and keep ID-backbone claims limited to Beauty or to domains where warm-strict is feasible.

## 6. Boundary

Do not mix the settings: warm setting is for ID-based backbones; cold setting is for content carrier / cold-start diagnostics. No DeepSeek API should be launched until the chosen route is clear.
