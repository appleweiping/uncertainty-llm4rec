# Day28 Cross-domain Medium Metric Design Report

## 1. Metric Repair

We no longer use HR@10 as a primary metric when the candidate pool has one positive plus five negatives. In that setting each user has 6 candidates, so top-10 covers the entire pool and HR@10 is trivial. HR@10 can be retained only with an explicit triviality flag.

## 2. medium_5neg_2000

The `medium_5neg_2000` continuity split is stored as `data/processed/amazon_*_medium_5neg/` and mirrored by the default `data/processed/amazon_*_medium/` alias. It has 2000 users per domain and 1 positive + 5 negatives per valid/test user. It remains useful for calibration, NDCG, and MRR continuity, but HR@10 must not be interpreted.

## 3. medium_20neg_500

The `medium_20neg_500` split is stored as `data/processed/amazon_*_medium_20neg/`. It has 500 users per domain and 1 positive + 20 negatives per valid/test user. It is a low-cost ranking smoke setting where HR@10 is non-trivial.

## 4. medium_20neg_2000

The `medium_20neg_2000` split is stored as `data/processed/amazon_*_medium_20neg_2000/`. It has 2000 users per domain and 21 candidates per valid/test user. This is the preferred formal cross-domain medium benchmark because both user count and candidate-pool size are large enough for HR@1/3/10, NDCG@3/5/10, and MRR.

## 5. Cost Comparison

`medium_20neg_500` costs about `1.80x` Beauty Day9 rows per domain. `medium_20neg_2000` costs about `7.19x` Beauty Day9 rows per domain. Day29 should not run all domains at once.

### 5neg and 20neg-500

| domain | variant | medium_users | valid_rows | test_rows | total_api_rows | relative_to_beauty_day9 | recommended_day29_mode | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movies | medium_5neg | 2000 | 12000 | 12000 | 24000 | 2.0555 | do_not_use_hr10_as_primary | 5-negative medium keeps more users at controlled cost, but HR@10 is trivial because each user has 6 candidates. |
| books | medium_5neg | 2000 | 12000 | 12000 | 24000 | 2.0555 | do_not_use_hr10_as_primary | 5-negative medium keeps more users at controlled cost, but HR@10 is trivial because each user has 6 candidates. |
| electronics | medium_5neg | 2000 | 12000 | 12000 | 24000 | 2.0555 | do_not_use_hr10_as_primary | 5-negative medium keeps more users at controlled cost, but HR@10 is trivial because each user has 6 candidates. |
| movies | medium_20neg | 500 | 10500 | 10500 | 21000 | 1.7986 | movies_medium_20neg_first | 20-negative medium gives non-trivial HR@10 with API rows close to the 5neg medium. |
| books | medium_20neg | 500 | 10500 | 10500 | 21000 | 1.7986 | books_medium_20neg_later | 20-negative medium gives non-trivial HR@10 with API rows close to the 5neg medium. |
| electronics | medium_20neg | 500 | 10500 | 10500 | 21000 | 1.7986 | electronics_medium_20neg_later | 20-negative medium gives non-trivial HR@10 with API rows close to the 5neg medium. |

### 20neg-2000

| domain | medium_users | negatives_per_positive | valid_rows | test_rows | total_api_rows | relative_to_beauty_day9 | recommended_run_mode | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movies | 2000 | 20 | 42000 | 42000 | 84000 | 7.1942 | movies_medium_20neg_2000_first | medium_20neg_2000 is the formal cross-domain medium benchmark: enough users and 21 candidates per user; run one domain at a time. |
| books | 2000 | 20 | 42000 | 42000 | 84000 | 7.1942 | books_medium_20neg_2000_hold | medium_20neg_2000 is the formal cross-domain medium benchmark: enough users and 21 candidates per user; run one domain at a time. |
| electronics | 2000 | 20 | 42000 | 42000 | 84000 | 7.1942 | electronics_medium_20neg_2000_hold | medium_20neg_2000 is the formal cross-domain medium benchmark: enough users and 21 candidates per user; run one domain at a time. |

## 6. Schema Validation: medium_20neg_500

| domain | processed_path | split | num_rows | num_users | num_items | positive_rows | negative_rows | avg_candidates_per_user | min_candidates_per_user | max_candidates_per_user | has_user_id | has_history | has_candidate_item_id | has_candidate_text | has_candidate_title | has_label | schema_compatible_with_beauty | missing_fields | notes | benchmark_variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movies | data\processed\amazon_movies_medium_20neg | train | 34524 | 324 | 32254 | 1644 | 32880 | 106.5556 | 21.0000 | 1491.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg |
| movies | data\processed\amazon_movies_medium_20neg | valid | 10500 | 500 | 10278 | 500 | 10000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg |
| movies | data\processed\amazon_movies_medium_20neg | test | 10500 | 500 | 10271 | 500 | 10000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg |
| books | data\processed\amazon_books_medium_20neg | train | 41811 | 330 | 40881 | 1991 | 39820 | 126.7000 | 21.0000 | 3339.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg |
| books | data\processed\amazon_books_medium_20neg | valid | 10500 | 500 | 10446 | 500 | 10000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg |
| books | data\processed\amazon_books_medium_20neg | test | 10500 | 500 | 10446 | 500 | 10000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg |
| electronics | data\processed\amazon_electronics_medium_20neg | train | 31353 | 315 | 30413 | 1493 | 29860 | 99.5333 | 21.0000 | 1008.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg |
| electronics | data\processed\amazon_electronics_medium_20neg | valid | 10500 | 500 | 10389 | 500 | 10000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg |
| electronics | data\processed\amazon_electronics_medium_20neg | test | 10500 | 500 | 10388 | 500 | 10000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg |

## 7. Schema Validation: medium_20neg_2000

| domain | processed_path | split | num_rows | num_users | num_items | positive_rows | negative_rows | avg_candidates_per_user | min_candidates_per_user | max_candidates_per_user | has_user_id | has_history | has_candidate_item_id | has_candidate_text | has_candidate_title | has_label | schema_compatible_with_beauty | missing_fields | notes | benchmark_variant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movies | data\processed\amazon_movies_medium_20neg_2000 | train | 158634 | 1296 | 118193 | 7554 | 151080 | 122.4028 | 21.0000 | 4704.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg_2000 |
| movies | data\processed\amazon_movies_medium_20neg_2000 | valid | 42000 | 2000 | 38589 | 2000 | 40000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg_2000 |
| movies | data\processed\amazon_movies_medium_20neg_2000 | test | 42000 | 2000 | 38671 | 2000 | 40000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg_2000 |
| books | data\processed\amazon_books_medium_20neg_2000 | train | 175308 | 1278 | 160106 | 8348 | 166960 | 137.1737 | 21.0000 | 5103.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg_2000 |
| books | data\processed\amazon_books_medium_20neg_2000 | valid | 42000 | 2000 | 41057 | 2000 | 40000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg_2000 |
| books | data\processed\amazon_books_medium_20neg_2000 | test | 42000 | 2000 | 41021 | 2000 | 40000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg_2000 |
| electronics | data\processed\amazon_electronics_medium_20neg_2000 | train | 126294 | 1242 | 112079 | 6014 | 120280 | 101.6860 | 21.0000 | 4662.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg_2000 |
| electronics | data\processed\amazon_electronics_medium_20neg_2000 | valid | 42000 | 2000 | 40161 | 2000 | 40000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg_2000 |
| electronics | data\processed\amazon_electronics_medium_20neg_2000 | test | 42000 | 2000 | 40207 | 2000 | 40000 | 21.0000 | 21.0000 | 21.0000 | True | True | True | True | True | True | True |  | medium split from regular processed domain | medium_20neg_2000 |

## 8. Recommendation

Day29 should run Movies `medium_20neg_2000` relevance evidence first. If cost or time pressure is high, run Movies `medium_20neg_500` first as a smoke test, but the final cross-domain main result should prioritize the 2000-user 20-negative benchmark. Books and Electronics are constructed and validated but should wait until Movies confirms the pipeline.
