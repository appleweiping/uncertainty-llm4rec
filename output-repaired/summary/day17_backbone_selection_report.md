# Day17 Healthy Ranking Backbone Selection

Day16 showed that BPR-MF is not a healthy final external backbone for the current Beauty candidate pool: it is user-id based, has high fallback, and remains weaker than popularity on the 100-user slice. Day17 therefore switches to a history-based sequential backbone.

Candidate options reviewed:

- LLMEmb SASRec: code exists in `external/LLMEmb/models/SASRec.py`, but the official repo still depends on missing handled data, LLM/SRS embeddings, and checkpoints for its full pipeline. It remains useful as a design reference but is not the fastest honest plug-in backbone.
- Minimal SASRec-style sequence encoder: can be trained directly from `data/processed/amazon_beauty/train.jsonl` using mapped history titles and positive next items. It does not depend on user-id embeddings, so it should reduce the unknown-user fallback that broke BPR-MF.
- Sequential ItemKNN/co-occurrence: lower-cost fallback, but less faithful to a neural sequential ranking backbone.

Selected Day17 backbone: minimal SASRec-style sequence encoder.

Why it is healthier than BPR-MF:

- It scores candidates from user history sequence rather than from a learned user-id embedding.
- It can score users that never appeared as user IDs in the train positives, as long as their history titles map to item IDs.
- It exports full candidate-pool scores for the same Day9/Day10 Beauty users, enabling fair Scheme 4 plug-in reranking.

Expected risk:

- Candidate items unseen in train remain cold and require explicit fallback.
- History title-to-item mapping may be imperfect.
- This is a 100-user smoke test, not a final SOTA claim.
