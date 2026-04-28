# Framework-Observation-Day2 Generative Recommendation Smoke Report

Status: observation only. This is not training, CEP, evidence decomposition, or a continuation of yes/no confidence prompting.

## Setup

- Task: Beauty candidate-grounded title generation.
- Model: base Qwen3-8B first; LoRA remains optional.
- Output schema: `recommended_title` plus raw verbalized `confidence`.
- Evaluation: generated title is grounded back to the 6-item candidate pool and then to the catalog when needed.

## Diagnostics

| split | num_users | parse_success_rate | schema_valid_rate | valid_candidate_title_rate | catalog_match_rate | hallucination_rate | HR@1 | MRR | NDCG@3 | NDCG@5 | NDCG@10 | target_rank_mean | confidence_mean | confidence_std | confidence_unique_count | confidence_ge_0.9_rate | ECE_for_generation_correctness | Brier_for_generation_correctness | AUROC_for_generation_correctness | high_conf_wrong_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | 100 | 1.0 | 1.0 | 0.12 | 0.13 | 0.87 | 0.17 | 0.395 | 0.31202086796428946 | 0.47552674102587644 | 0.5396440347053204 | 3.67 | 0.123 | 0.31823104813955533 | 3 | 0.13 | 0.04299999999999999 | 0.0444 | 0.9782608695652174 | 0.05 |
| test | 100 | 1.0 | 1.0 | 0.12 | 0.13 | 0.87 | 0.13 | 0.3518333333333333 | 0.25940227289286033 | 0.4022507930675546 | 0.505550877328881 | 4.03 | 0.121 | 0.3135426605742829 | 6 | 0.11 | 0.08099999999999999 | 0.07894999999999999 | 0.9583333333333334 | 0.08 |

## Calibration

| split | score_type | fit_split | ECE | Brier | AUROC | note |
| --- | --- | --- | --- | --- | --- | --- |
| valid | raw_verbalized_confidence | none | 0.04299999999999999 | 0.0444 | 0.9782608695652174 | diagnostic_only |
| test | raw_verbalized_confidence | none | 0.08099999999999999 | 0.07894999999999999 | 0.9583333333333334 |  |
| valid | calibrated_verbalized_confidence | valid | 0.0 | 0.03076923076923077 | 0.9728260869565217 | valid_fit_test_evaluate |
| test | calibrated_verbalized_confidence | valid | 0.046092307692307696 | 0.04326172781065089 | 0.9466145833333334 | valid_fit_test_evaluate |

## Interpretation Template

- If candidate-grounded title validity and HR@1 are reasonable, Day3 can expand to valid/test 500 or Beauty full.
- If raw verbalized confidence collapses again, Day3 should add generation logprob, title retrieval margin, and title self-consistency agreement.
- If hallucination is high, keep candidate-grounded generation and defer open-title full runs.
- If candidate-grounded generation is strong, later evidence observation or CEP can be considered, but Day2 itself is still observation.
