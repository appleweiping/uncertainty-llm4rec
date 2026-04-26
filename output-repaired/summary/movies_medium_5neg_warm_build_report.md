# Movies medium_5neg_warm Build Report

## 1. Purpose

This warm split is created separately from the existing cold-style `data/processed/amazon_movies_medium_5neg/`. The original directory is not overwritten. Warm negatives are sampled from `train_candidate_vocab - user_seen_items`, so ID-based backbones can evaluate mostly warm negative candidates.

## 2. Output

Output directory: `data\processed\amazon_movies_medium_5neg_warm`.

Rows: train `45324`, valid `12000`, test `12000`.

## 3. Schema

Schema validation is saved to `output-repaired/summary/movies_medium_5neg_warm_schema_validation.csv`. The split remains Beauty-compatible and uses 1 positive + 5 negatives per user for valid/test.

## 4. Cold Rate

Using `train_backbone_vocab`:

- Valid positive cold rate: `0.6025`; valid negative cold rate: `0.0000`; valid all-candidate cold rate: `0.1004`.
- Test positive cold rate: `0.6235`; test negative cold rate: `0.0000`; test all-candidate cold rate: `0.1039`.

Warm negative sampling sharply reduces negative cold rate. Any remaining positive cold rate is a chronological/domain cold-start limitation and should be reported separately before running ID-based backbones.

## 5. Day35 Recommendation

If positive cold rate is acceptable for the intended claim, Day35 can run Movies warm relevance evidence and then evaluate ID-based SASRec/GRU4Rec/Bert4Rec. If positive cold rate remains too high, Movies ID-backbone evaluation should be marked as cold-start limited rather than forced.
