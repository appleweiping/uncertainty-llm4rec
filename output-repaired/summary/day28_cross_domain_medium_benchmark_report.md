# Day28 Cross-domain Medium Benchmark Report

## 1. Why Not Direct Full

Regular Books/Electronics/Movies are large. Direct full evidence inference would mix data repair, API cost, and experimental validation all at once. Day28 therefore builds a standardized medium benchmark first.

## 2. Why Not Old Small

The old small domains are useful for quick debugging, but they are too small to carry the final cross-domain claim. The medium benchmark is sampled from regular processed domain interactions and items, so it is not a toy small split.

## 3. Medium Benchmark Definition

- Source: regular `data/processed/amazon_*` interactions/items/users.
- Sampling: eligible users have at least 3 unique chronological interactions.
- User sample: reservoir sampling with seed `42`, then deterministic user-id sorting.
- Split: user-level chronological leave-one-out; last interaction is test, second-to-last is valid, earlier interactions are train history.
- Negative sampling: seed `42`, 5 negatives per positive, excluding items already interacted with by the user.
- Schema: Beauty-compatible pointwise JSONL with `user_id`, `history`, `candidate_item_id`, `candidate_title`, `candidate_text`, `label`, `target_popularity_group`, and `timestamp`.

## 4. Eligible Users And Medium Users

| domain | eligible_users | medium_users | train_rows | valid_rows | test_rows | status | output_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| movies | 1174877 | 2000 | 45324 | 12000 | 12000 | ready | data\processed\amazon_movies_medium |
| books | 1711594 | 2000 | 50088 | 12000 | 12000 | ready | data\processed\amazon_books_medium |
| electronics | 3007066 | 2000 | 36084 | 12000 | 12000 | ready | data\processed\amazon_electronics_medium |

## 5. Schema Validation

| domain | processed_path | split | num_rows | num_users | num_items | positive_rows | negative_rows | avg_candidates_per_user | min_candidates_per_user | max_candidates_per_user | has_user_id | has_history | has_candidate_item_id | has_candidate_text | has_candidate_title | has_label | schema_compatible_with_beauty | missing_fields | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movies | data\processed\amazon_movies_medium | train | 45324 | 1296 | 40848 | 7554 | 37770 | 34.9722 | 6.0000 | 1344.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain |
| movies | data\processed\amazon_movies_medium | valid | 12000 | 2000 | 11549 | 2000 | 10000 | 6.0000 | 6.0000 | 6.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain |
| movies | data\processed\amazon_movies_medium | test | 12000 | 2000 | 11480 | 2000 | 10000 | 6.0000 | 6.0000 | 6.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain |
| books | data\processed\amazon_books_medium | train | 50088 | 1278 | 48541 | 8348 | 41740 | 39.1925 | 6.0000 | 1458.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain |
| books | data\processed\amazon_books_medium | valid | 12000 | 2000 | 11895 | 2000 | 10000 | 6.0000 | 6.0000 | 6.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain |
| books | data\processed\amazon_books_medium | test | 12000 | 2000 | 11863 | 2000 | 10000 | 6.0000 | 6.0000 | 6.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain |
| electronics | data\processed\amazon_electronics_medium | train | 36084 | 1242 | 34297 | 6014 | 30070 | 29.0531 | 6.0000 | 1332.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain |
| electronics | data\processed\amazon_electronics_medium | valid | 12000 | 2000 | 11700 | 2000 | 10000 | 6.0000 | 6.0000 | 6.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain |
| electronics | data\processed\amazon_electronics_medium | test | 12000 | 2000 | 11733 | 2000 | 10000 | 6.0000 | 6.0000 | 6.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain |

## 6. API Cost Estimate

| domain | medium_users | valid_rows | test_rows | total_api_rows | relative_to_beauty_day9 | recommended_day29_mode | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| movies | 2000 | 12000 | 12000 | 24000 | 2.0555 | movies_medium_first | Movies is the recommended first cross-domain medium run. |
| books | 2000 | 12000 | 12000 | 24000 | 2.0555 | books_medium_first | Books medium is ready after Movies. |
| electronics | 2000 | 12000 | 12000 | 24000 | 2.0555 | electronics_medium_first | Electronics medium is ready but recommended after Movies/Books as a harder/noisier domain. |

## 7. Day29 Recommendation

Run **Movies medium relevance evidence** first. Movies is the preferred first cross-domain run because it already had partial pointwise traces and now has a repaired medium split from the regular domain. Books and Electronics medium splits are built and validated, but should follow after Movies.
