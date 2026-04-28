# Framework-Observation-Day2c Label-First Generation Report

Status: observation only. This is not training, evidence, CEP, external API use, open-title generation, or a full run.

## Framing

Day2 exposed placeholder generation failure. Day2b fixed candidate-title validity but still had explanatory text and unusable confidence. Day2c tests whether label-first generation gives clean candidate-grounded output.

## Diagnostics

| split | num_users | parse_success_rate | schema_valid_rate | generation_valid_rate | label_valid_rate | title_matches_selected_label_rate | candidate_title_exact_match_rate | catalog_match_rate | matched_title_hit_rate | selected_label_hit_rate | hallucination_rate | placeholder_title_rate | explanatory_text_after_json_rate | selected_label_distribution | confidence_mean | confidence_std | confidence_unique_count | confidence_AUROC_for_hit | confidence_ECE_for_hit | confidence_Brier_for_hit | high_conf_wrong_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | 100 | 0.68 | 0.68 | 0.64 | 0.68 | 0.64 | 0.64 | 0.68 | 0.2 | 0.2 | 0.32 | 0.32 | 0.0 | {"A": 2, "B": 12, "C": 14, "D": 17, "E": 16, "F": 7, "INVALID": 32} | 0.5957 | 0.41291949578580084 | 6 | 0.749375 | 0.3957 | 0.366161 | 0.32 |
| test | 100 | 0.75 | 0.75 | 0.74 | 0.75 | 0.74 | 0.74 | 0.75 | 0.15 | 0.15 | 0.25 | 0.25 | 0.0 | {"A": 4, "B": 6, "C": 19, "D": 26, "E": 13, "F": 7, "INVALID": 25} | 0.6636 | 0.3873435684247255 | 5 | 0.6627450980392157 | 0.5136000000000001 | 0.4736 | 0.44 |

## Control Comparison

| method | prompt_version | output_schema | num_users | parse_success_rate | schema_valid_rate | generation_valid_rate | placeholder_title_rate | label_valid_rate | title_matches_selected_label_rate | candidate_title_exact_match_rate | catalog_match_rate | matched_title_hit_rate | hallucination_rate | explanatory_text_after_json_rate | confidence_mean | confidence_std | confidence_AUROC | confidence_ECE | interpretation | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_qwen_candidate_grounded | day2_placeholder_schema | recommended_title+confidence | 100 | 1.0 | 1.0 | 0.13 | NA | NA | NA | 0.12 | 0.13 | 0.13 | 0.87 | NA | 0.121 | 0.3135426605742829 | 0.9583333333333334 | 0.08099999999999999 | Day2 placeholder/schema failure; parse/schema rates are superficial. | do_not_full_run |
| base_qwen_candidate_grounded | day2b_no_placeholder_exact_title | recommended_title+confidence | 100 | 1.0 | 1.0 | 1.0 | 0.0 | NA | NA | 0.8 | 1.0 | 0.19 | 0.0 | 1.0 | 0.8187 | 0.025520775850275394 | 0.48862897985705 | 0.6356999999999999 | Day2b fixed validity but retained explanatory text and unusable confidence. | do_not_full_run |
| base_qwen_candidate_grounded | day2c_label_first | selected_label+recommended_title+confidence | 100 | 0.75 | 0.75 | 0.74 | 0.25 | 0.75 | 0.74 | 0.74 | 0.75 | 0.15 | 0.25 | 0.0 | 0.6636 | 0.3873435684247255 | 0.6627450980392157 | 0.5136000000000001 | Label-first succeeds if validity and title-label agreement approach 1.0 and explanatory text drops. | move_to_non_verbal_uncertainty |

## Interpretation Rules

- If generation validity and title-label agreement are near 1.0 and explanatory text falls, output control is clean.
- If matched-title hit rate remains around Day2b, base Qwen candidate-grounded recommendation ability is modest; do not overclaim.
- If confidence remains low-variance, AUROC near 0.5, or ECE high, raw verbalized confidence remains unusable.
- If confidence is unusable, move to selected-label logprob, title logprob, retrieval margin, self-consistency title agreement, or label-selection entropy.
