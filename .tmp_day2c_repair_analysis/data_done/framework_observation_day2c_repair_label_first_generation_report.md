# Framework-Observation-Day2c-Repair Label-First Generation Report

Status: observation only. This is not training, evidence, CEP, external API use, open-title generation, or a full run.

## Framing

Day2 exposed placeholder generation failure. Day2b fixed candidate-title validity but still had explanatory text and unusable confidence. Day2c fixed explanatory tails but failed due to long-title truncation. Day2c-repair tests whether increasing token budget and compact field order can make label-first title generation output-stable.

## Diagnostics

| split | num_users | parse_success_rate | schema_valid_rate | generation_valid_rate | label_valid_rate | title_matches_selected_label_rate | candidate_title_exact_match_rate | catalog_match_rate | matched_title_hit_rate | selected_label_hit_rate | hallucination_rate | placeholder_title_rate | json_truncation_rate | explanatory_text_after_json_rate | selected_label_distribution | confidence_mean | confidence_std | confidence_unique_count | confidence_AUROC_for_hit | confidence_ECE_for_hit | confidence_Brier_for_hit | high_conf_wrong_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | 100 | 1.0 | 1.0 | 0.92 | 1.0 | 0.92 | 0.93 | 1.0 | 0.31 | 0.31 | 0.0 | 0.0 | 0.0 | 1.0 | {"A": 2, "B": 18, "C": 20, "D": 28, "E": 26, "F": 6} | 0.8462999999999999 | 0.017643979143039137 | 5 | 0.5238429172510519 | 0.5363 | 0.5017349999999999 | 0.0 |
| test | 100 | 1.0 | 1.0 | 0.95 | 1.0 | 0.95 | 0.95 | 1.0 | 0.26 | 0.26 | 0.0 | 0.0 | 0.0 | 1.0 | {"A": 4, "B": 14, "C": 21, "D": 33, "E": 21, "F": 7} | 0.8458 | 0.014709180806557514 | 4 | 0.5675675675675675 | 0.5858 | 0.5335939999999999 | 0.0 |

## Control Comparison

| method | prompt_version | output_schema | max_new_tokens | field_order | num_users | parse_success_rate | schema_valid_rate | generation_valid_rate | placeholder_title_rate | json_truncation_rate | label_valid_rate | title_matches_selected_label_rate | candidate_title_exact_match_rate | catalog_match_rate | matched_title_hit_rate | hallucination_rate | explanatory_text_after_json_rate | confidence_mean | confidence_std | confidence_AUROC | confidence_ECE | interpretation | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_qwen_candidate_grounded | day2_placeholder_schema | recommended_title+confidence | NA | recommended_title,confidence | 100 | 1.0 | 1.0 | 0.13 | NA | NA | NA | NA | 0.12 | 0.13 | 0.13 | 0.87 | NA | 0.121 | 0.3135426605742829 | 0.9583333333333334 | 0.08099999999999999 | Day2 placeholder/schema failure; parse/schema rates are superficial. | do_not_full_run |
| base_qwen_candidate_grounded | day2b_no_placeholder_exact_title | recommended_title+confidence | 96 | recommended_title,confidence | 100 | 1.0 | 1.0 | 1.0 | 0.0 | NA | NA | NA | 0.8 | 1.0 | 0.19 | 0.0 | 1.0 | 0.8187 | 0.025520775850275394 | 0.48862897985705 | 0.6356999999999999 | Day2b fixed validity but retained explanatory text and unusable confidence. | do_not_full_run |
| base_qwen_candidate_grounded | day2c_repair_label_first_compact | selected_label+recommended_title+confidence | 160 | selected_label,confidence,recommended_title | 100 | 1.0 | 1.0 | 0.95 | 0.0 | 0.0 | 1.0 | 0.95 | 0.95 | 1.0 | 0.26 | 0.0 | 1.0 | 0.8458 | 0.014709180806557514 | 0.5675675675675675 | 0.5858 | Label-first succeeds if validity and title-label agreement approach 1.0 and explanatory text drops. | move_to_non_verbal_uncertainty |

## Interpretation Rules

- If generation validity and title-label agreement are near 1.0 and explanatory text falls, output control is clean.
- If matched-title hit rate remains around Day2b, base Qwen candidate-grounded recommendation ability is modest; do not overclaim.
- If confidence remains low-variance, AUROC near 0.5, or ECE high, raw verbalized confidence remains unusable.
- If confidence is unusable, move to selected-label logprob, title logprob, retrieval margin, self-consistency title agreement, or label-selection entropy.
- Day2d non-verbal uncertainty remains paused until parse/schema/generation/title-label validity are >= 0.95, JSON truncation <= 0.05, and explanatory text after JSON <= 0.05.
- If output control passes but matched-title hit rate remains 0.15-0.20, candidate-grounded generation is controllable but base Qwen recommendation choice is modest.
