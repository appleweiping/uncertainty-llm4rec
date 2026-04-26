# Movies medium_5neg Cold-Rate Diagnostic

## 1. Goal

This diagnostic separates candidate count from cold-candidate composition. The `5neg` setting controls how many negatives are sampled per positive. It does not by itself cause high cold rate. Cold rate is driven by train-unseen candidate coverage under the chosen sampling pool; 5neg controls candidate count, while the sampling pool determines warm/cold composition.

## 2. Train Vocabulary Definitions

The diagnostic reports three vocabularies:

- `train_candidate_vocab`: every `candidate_item_id` appearing in `train.jsonl`.
- `train_history_vocab`: item ids extracted from train `history` entries.
- `train_backbone_vocab`: union of the two. This is the main reference because ID-based backbones need item embeddings for both history and scored candidates.

## 3. Main Cold-Rate Result

Using `train_backbone_vocab`:

- Valid positive cold rate: `0.6025`; valid negative cold rate: `0.8408`; valid all-candidate cold rate: `0.8011`.
- Test positive cold rate: `0.6235`; test negative cold rate: `0.8417`; test all-candidate cold rate: `0.8053`.

Interpretation:

- Valid cause: both all-items negative sampling and cold future positives.
- Test cause: both all-items negative sampling and cold future positives.

Both negative cold rate and positive cold rate are high. This means the current Movies medium_5neg is not merely drawing cold negatives; the chronological split also places many valid/test positive items outside the train backbone vocabulary.

## 4. Why ID-Based Backbones Break

SASRec, GRU4Rec, and Bert4Rec use item-id embeddings. They can only score items that have train-time embeddings in the backbone vocabulary. When valid/test candidates are mostly train-unseen, these models must fallback or produce unreliable scores for many rows. This explains the Day32 Movies SASRec fallback problem without blaming CEP.

## 5. Setting Interpretation

The current `data/processed/amazon_movies_medium_5neg/` should be treated as a cold-style sampling setting because negatives were sampled from the regular domain all-item pool, and many future positives are also train-unseen. Do not overwrite this directory.

## 6. Recommended Split Strategy

Keep two separate Movies settings:

- `movies_medium_5neg_warm`: sample negatives from `train_seen_items - user_seen_items`. Use this for ID-based backbone plug-in evaluation with SASRec/GRU4Rec/Bert4Rec.
- `movies_medium_5neg_cold`: sample negatives from all items and allow cold candidates. Use this for TF-IDF/BM25/content carrier + CEP cold-start diagnostics.

TF-IDF/BM25 should be described as a cold-aware content carrier or diagnostic backbone, not as a SOTA recommender. It is useful here because it can score candidate_title/candidate_text without requiring train item-id embeddings.
