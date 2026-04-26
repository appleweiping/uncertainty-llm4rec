# Day34 Movies Cold-Style Content Carrier Report

## 1. Setting

This run uses the existing `data/processed/amazon_movies_medium_5neg/` as a cold-style sampling setting. It does not call DeepSeek API and reuses the existing Movies CEP evidence. HR@10 is retained in tables but marked trivial because each user has 6 candidates.

## 2. Backbone

The content carriers are TF-IDF cosine similarity and BM25-style similarity between user history text and candidate title/text. They do not depend on train item-id embeddings, so they can score cold candidates. They are diagnostic cold-aware content carriers, not SOTA recommender backbones.

## 3. Join

Join diagnostics are saved to `day34_movies_cold_content_carrier_join_diagnostics.csv`. Fallback is expected to be zero because the carrier scores from text rather than item-id vocab.

## 4. Plug-in Result

- `tfidf_cosine` best `D_Backbone_plus_calibrated_relevance_plus_evidence_risk`: NDCG@10 `0.7378`, MRR `0.6527`, HR@1 `0.4780`, HR@3 `0.7680`.
- `bm25` best `D_Backbone_plus_calibrated_relevance_plus_evidence_risk`: NDCG@10 `0.6891`, MRR `0.5879`, HR@1 `0.3750`, HR@3 `0.7415`.

## 5. Interpretation

This test asks whether CEP can plug into a cold-aware content carrier when ID-based sequential backbones are invalid. It should not be compared as a strong external recommender claim. The proper ID-backbone evaluation should use the separately constructed warm split.
