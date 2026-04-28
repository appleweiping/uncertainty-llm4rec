# Framework-Observation-Day2b Generative Recommendation Prompt/Parser Repair Report

Status: observation only. This is not training, CEP, evidence decomposition, or a continuation of yes/no confidence prompting.

## Setup

- Task: Beauty candidate-grounded title generation.
- Model: base Qwen3-8B first; LoRA remains optional.
- Output schema: `recommended_title` plus raw verbalized `confidence`.
- Evaluation: generated title is grounded back to the 6-item candidate pool and then to the catalog when needed.
- Placeholder titles such as `...`, empty strings, `N/A`, `unknown`, and `none` are generation-invalid.

## Day2b Interpretation

- Day2 exposed a placeholder/schema-following failure: parse/schema success can be superficial.
- Day2b tests whether removing placeholder examples and forcing exact candidate-title copying fixes output control.
- Main metrics are generation validity, exact candidate-title matching, matched-title hit rate, hallucination, and placeholder rate.

## Diagnostics

| split | num_users | parse_success_rate | schema_valid_rate | generation_valid_rate | placeholder_title_rate | empty_title_rate | invalid_title_rate | explanatory_text_after_json_rate | candidate_title_exact_match_rate | valid_candidate_title_rate | catalog_match_rate | matched_title_hit_rate | matched_title_hit_rate_given_generation_valid | hallucination_rate | HR@1 | MRR | NDCG@3 | NDCG@5 | NDCG@10 | target_rank_mean | confidence_mean | confidence_std | confidence_unique_count | confidence_ge_0.9_rate | ECE_for_generation_correctness | Brier_for_generation_correctness | AUROC_for_generation_correctness | high_conf_wrong_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | 100 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.8 | 0.95 | 1.0 | 0.25 | 0.25 | 0.0 | 0.26 | 0.4683333333333334 | 0.40833016550000406 | 0.5352663358769282 | 0.595821557685292 | 3.31 | 0.8239 | 0.029492202359267777 | 4 | 0.0 | 0.5739 | 0.5160809999999999 | 0.4952 | 0.0 |
| test | 100 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.8 | 0.97 | 1.0 | 0.19 | 0.19 | 0.0 | 0.2 | 0.43 | 0.3785673556428623 | 0.49900557289710995 | 0.5666849384476342 | 3.49 | 0.8187 | 0.025520775850275394 | 4 | 0.0 | 0.6356999999999999 | 0.5515209999999999 | 0.48862897985705 | 0.0 |

## Calibration

| split | score_type | fit_split | ECE | Brier | AUROC | note |
| --- | --- | --- | --- | --- | --- | --- |
| valid | raw_verbalized_confidence | none | 0.5739 | 0.5160809999999999 | 0.4952 | diagnostic_only |
| test | raw_verbalized_confidence | none | 0.6356999999999999 | 0.5515209999999999 | 0.48862897985705 |  |
| valid | calibrated_verbalized_confidence | valid | 0.0 | 0.18556701030927833 | 0.52 | valid_fit_test_evaluate |
| test | calibrated_verbalized_confidence | valid | 0.07484536082474225 | 0.1603209692847274 | 0.4983755685510071 | valid_fit_test_evaluate |

## Interpretation Template

- If candidate-grounded title validity and HR@1 are reasonable, Day3 can expand to valid/test 500 or Beauty full.
- If raw verbalized confidence collapses again, Day3 should add generation logprob, title retrieval margin, and title self-consistency agreement.
- If hallucination is high, keep candidate-grounded generation and defer open-title full runs.
- If placeholder outputs persist, switch to the Day2c label-first generation fallback before expanding sample size.
- If candidate-grounded generation is strong, later evidence observation or CEP can be considered, but Day2 itself is still observation.
