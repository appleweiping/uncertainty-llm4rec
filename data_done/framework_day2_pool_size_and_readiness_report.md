# Framework-Day2 Pool-Size and Readiness Report

## 1. Day1 Recap

Framework-Day1 created a clean `data_done/` foundation for Beauty, Books, Electronics, and Movies using user_min4, chronological leave-one-out, max 10,000 users/domain, seed=42, and warm negative sampling.

## 2. Why Keep 5neg

The original `valid.jsonl` and `test.jsonl` remain the 5neg continuity split. It is low cost, aligned with the Beauty observation-stage candidate-pool setting, and useful for LoRA/evidence-generator data scaffolding. HR@10 is trivial because each user has only six candidates, so claims should use NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.

## 3. Why Add 20neg

`eval_20neg/` adds 20 negatives per positive, giving each user 21 candidates per split. HR@10 is no longer trivial, making this split better for formal ranking/backbone evaluation. It is not an API launch plan: Books/Electronics/Movies each have 420,000 valid+test rows in 20neg, so DeepSeek inference requires explicit budget confirmation.

## 4. Four-Domain 5neg vs 20neg Cost

| domain | candidate_pool_setting | users | valid_rows | test_rows | candidates_per_user | negatives_per_positive | hr10_trivial_flag | estimated_api_rows_valid_test | relative_to_beauty_day9 | recommended_usage | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beauty | 5neg | 622 | 3732 | 3732 | 6 | 5 | True | 7464 | 0.6392600205549845 | low_cost_CEP_continuity, lora_training_candidate, not_for_HR10_claim | Continuity split; HR@10 is trivial. |
| beauty | 20neg | 622 | 13062 | 13062 | 21 | 20 | False | 26124 | 2.237410071942446 | ranking_eval, HR10_valid, high_api_cost | Do not launch DeepSeek without explicit budget confirmation. |
| books | 5neg | 10000 | 60000 | 60000 | 6 | 5 | True | 120000 | 10.27749229188078 | low_cost_CEP_continuity, lora_training_candidate, not_for_HR10_claim | Continuity split; HR@10 is trivial. |
| books | 20neg | 10000 | 210000 | 210000 | 21 | 20 | False | 420000 | 35.97122302158273 | ranking_eval, HR10_valid, high_api_cost | Do not launch DeepSeek without explicit budget confirmation. |
| electronics | 5neg | 10000 | 60000 | 60000 | 6 | 5 | True | 120000 | 10.27749229188078 | low_cost_CEP_continuity, lora_training_candidate, not_for_HR10_claim | Continuity split; HR@10 is trivial. |
| electronics | 20neg | 10000 | 210000 | 210000 | 21 | 20 | False | 420000 | 35.97122302158273 | ranking_eval, HR10_valid, high_api_cost | Do not launch DeepSeek without explicit budget confirmation. |
| movies | 5neg | 10000 | 60000 | 60000 | 6 | 5 | True | 120000 | 10.27749229188078 | low_cost_CEP_continuity, lora_training_candidate, not_for_HR10_claim | Continuity split; HR@10 is trivial. |
| movies | 20neg | 10000 | 210000 | 210000 | 21 | 20 | False | 420000 | 35.97122302158273 | ranking_eval, HR10_valid, high_api_cost | Do not launch DeepSeek without explicit budget confirmation. |

## 5. Text Fallback Coverage

| domain | candidate_pool_setting | split | candidate_text_missing_rate | candidate_text_fallback_rate | history_text_missing_rate | history_text_fallback_rate | text_coverage_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beauty | 5neg | valid | 0.0 | 0.0 | 0.0 | 0.0 | ok | metadata text available |
| beauty | 5neg | test | 0.0 | 0.0 | 0.0 | 0.0 | ok | metadata text available |
| beauty | 20neg | valid | 0.0 | 0.0 | 0.0 | 0.0 | ok | metadata text available |
| beauty | 20neg | test | 0.0 | 0.0 | 0.0 | 0.0 | ok | metadata text available |
| books | 5neg | valid | 0.0 | 0.0 | 0.0 | 0.0 | ok | metadata text available |
| books | 5neg | test | 0.0 | 0.0 | 0.0 | 0.0 | ok | metadata text available |
| books | 20neg | valid | 0.0 | 0.0 | 0.0 | 0.0 | ok | metadata text available |
| books | 20neg | test | 0.0 | 0.0 | 0.0 | 0.0 | ok | metadata text available |
| electronics | 5neg | valid | 1.6666666666666667e-05 | 1.6666666666666667e-05 | 1.827318410232983e-05 | 1.827318410232983e-05 | fallback_present | deterministic missing-metadata fallback is present; do not treat fallback text as semantic description |
| electronics | 5neg | test | 3.3333333333333335e-05 | 3.3333333333333335e-05 | 1.827318410232983e-05 | 1.827318410232983e-05 | fallback_present | deterministic missing-metadata fallback is present; do not treat fallback text as semantic description |
| electronics | 20neg | valid | 3.809523809523809e-05 | 3.809523809523809e-05 | 1.827318410232983e-05 | 1.827318410232983e-05 | fallback_present | deterministic missing-metadata fallback is present; do not treat fallback text as semantic description |
| electronics | 20neg | test | 1.4285714285714284e-05 | 1.4285714285714284e-05 | 1.827318410232983e-05 | 1.827318410232983e-05 | fallback_present | deterministic missing-metadata fallback is present; do not treat fallback text as semantic description |
| movies | 5neg | valid | 0.4114666666666666 | 0.4114666666666666 | 0.4525950936743658 | 0.4525950936743658 | fallback_present | deterministic missing-metadata fallback is present; do not treat fallback text as semantic description |
| movies | 5neg | test | 0.4104333333333333 | 0.4104333333333333 | 0.4525950936743658 | 0.4525950936743658 | fallback_present | deterministic missing-metadata fallback is present; do not treat fallback text as semantic description |
| movies | 20neg | valid | 0.3902571428571428 | 0.3902571428571428 | 0.4525950936743658 | 0.4525950936743658 | fallback_present | deterministic missing-metadata fallback is present; do not treat fallback text as semantic description |
| movies | 20neg | test | 0.3905142857142857 | 0.3905142857142857 | 0.4525950936743658 | 0.4525950936743658 | fallback_present | deterministic missing-metadata fallback is present; do not treat fallback text as semantic description |

## 6. Cold-Rate Diagnostics

Each `eval_20neg/cold_rate_diagnostics.csv` reports positive, negative, and all-candidate cold rates against train vocab. Warm negative sampling should keep negative cold rate near zero. Positive cold can remain high due to chronological held-out positives, so ID-based backbone results should be marked caution when positive cold rate exceeds 0.2.

## 7. Recommended Next Step

- Do not directly run DeepSeek on all three large-domain 20neg splits.
- Framework-Day3 can use 5neg for Qwen-LoRA training data scaffold and evidence-generator pair design.
- Alternatively, run Beauty `eval_20neg` local backbone ranking sanity first.
- Run 20neg DeepSeek evidence only one domain at a time after explicit budget approval.