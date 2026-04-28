# Framework-Observation-Day2c-Repair-Stop Label-First Generation Report

Status: observation only. This is not training, evidence, CEP, external API use, open-title generation, or a full run.

## Framing

Day2 exposed placeholder generation failure. Day2b fixed candidate-title validity but still had explanatory text and unusable confidence. Day2c fixed explanatory tails but failed due to long-title truncation. Day2c-repair fixed truncation but reintroduced explanatory tails. Day2c-repair-stop tests whether stop sequences or first-JSON extraction can make label-first candidate-grounded generation evaluable. This is an output-control repair, not a method contribution.

## Diagnostics

| split | num_users | parse_success_rate | first_json_parse_success_rate | schema_valid_rate | generation_valid_rate | label_valid_rate | title_matches_selected_label_rate | candidate_title_exact_match_rate | catalog_match_rate | matched_title_hit_rate | selected_label_hit_rate | hallucination_rate | placeholder_title_rate | json_truncation_rate | raw_ends_after_json_rate | raw_had_explanatory_tail_rate | explanatory_text_after_json_rate | selected_label_distribution | confidence_mean | confidence_std | confidence_unique_count | confidence_AUROC_for_hit | confidence_ECE_for_hit | confidence_Brier_for_hit | high_conf_wrong_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | 100 | 1.0 | 1.0 | 1.0 | 0.92 | 1.0 | 0.92 | 0.93 | 1.0 | 0.3 | 0.3 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | {"A": 2, "B": 18, "C": 19, "D": 29, "E": 26, "F": 6} | 0.8466 | 0.017448209077151732 | 5 | 0.5161904761904762 | 0.5466 | 0.5092359999999999 | 0.0 |
| test | 100 | 1.0 | 1.0 | 1.0 | 0.95 | 1.0 | 0.95 | 0.95 | 1.0 | 0.26 | 0.26 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | {"A": 5, "B": 14, "C": 21, "D": 32, "E": 21, "F": 7} | 0.8452 | 0.018082035283673126 | 5 | 0.5563929313929314 | 0.5852 | 0.5344899999999999 | 0.0 |

## Control Comparison

| method | prompt_version | output_schema | max_new_tokens | field_order | stop_strategy | first_json_extraction | num_users | parse_success_rate | first_json_parse_success_rate | schema_valid_rate | generation_valid_rate | placeholder_title_rate | json_truncation_rate | label_valid_rate | title_matches_selected_label_rate | candidate_title_exact_match_rate | catalog_match_rate | matched_title_hit_rate | hallucination_rate | raw_ends_after_json_rate | raw_had_explanatory_tail_rate | explanatory_text_after_json_rate | confidence_mean | confidence_std | confidence_AUROC | confidence_ECE | interpretation | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_qwen_candidate_grounded | day2_placeholder_schema | recommended_title+confidence | NA | recommended_title,confidence | none | yes | 100 | 1.0 | 1.0 | 1.0 | 0.13 | NA | NA | NA | NA | 0.12 | 0.13 | 0.13 | 0.87 | NA | NA | NA | 0.121 | 0.3135426605742829 | 0.9583333333333334 | 0.08099999999999999 | Day2 placeholder/schema failure; parse/schema rates are superficial. | do_not_full_run |
| base_qwen_candidate_grounded | day2b_no_placeholder_exact_title | recommended_title+confidence | 96 | recommended_title,confidence | none | yes | 100 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | NA | NA | NA | 0.8 | 1.0 | 0.19 | 0.0 | NA | 1.0 | 1.0 | 0.8187 | 0.025520775850275394 | 0.48862897985705 | 0.6356999999999999 | Day2b fixed validity but retained explanatory text and unusable confidence. | do_not_full_run |
| base_qwen_candidate_grounded | day2c_label_first_64_token | selected_label+recommended_title+confidence | 64 | selected_label,recommended_title,confidence | stop_after_newline | yes | 100 | 0.75 | 0.75 | 0.75 | 0.74 | 0.25 | 0.0 | NA | NA | 0.74 | 0.75 | 0.15 | 0.25 | NA | 0.0 | 0.0 | 0.6636 | 0.3873435684247255 | 0.6627450980392157 | 0.5136000000000001 | Day2c fixed explanatory tails but introduced JSON truncation from short token budget. | do_not_full_run |
| base_qwen_candidate_grounded | day2c_repair_label_first_compact | selected_label+recommended_title+confidence | 160 | selected_label,confidence,recommended_title | none | yes | 100 | 1.0 | 1.0 | 1.0 | 0.95 | 0.0 | 0.0 | NA | NA | 0.95 | 1.0 | 0.26 | 0.0 | NA | 1.0 | 1.0 | 0.8458 | 0.014709180806557514 | 0.5675675675675675 | 0.5858 | Day2c-repair fixed truncation but reintroduced explanatory tails. | do_not_full_run |
| base_qwen_candidate_grounded | day2c_repair_stop_label_first_compact | selected_label+recommended_title+confidence | 160 | selected_label,confidence,recommended_title | vllm_stop_variants_newline_okay_lets_blank_eos | yes | 100 | 1.0 | 1.0 | 1.0 | 0.95 | 0.0 | 0.0 | 1.0 | 0.95 | 0.95 | 1.0 | 0.26 | 0.0 | 1.0 | 0.0 | 0.0 | 0.8452 | 0.018082035283673126 | 0.5563929313929314 | 0.5852 | Label-first succeeds if validity and title-label agreement approach 1.0 and explanatory text drops. | move_to_non_verbal_uncertainty |

## Current Decision

Candidate-grounded generative output is raw-clean and evaluable.
Raw verbalized confidence remains overconfident and weakly informative for recommendation correctness.

## Interpretation Rules

- If generation validity and title-label agreement are near 1.0 and explanatory text falls, output control is clean.
- If matched-title hit rate remains around Day2b, base Qwen candidate-grounded recommendation ability is modest; do not overclaim.
- If confidence remains low-variance, AUROC near 0.5, or ECE high, raw verbalized confidence remains unusable.
- If confidence is unusable, move to selected-label logprob, title logprob, retrieval margin, self-consistency title agreement, or label-selection entropy only after output control is stable.
- Day2d non-verbal uncertainty remains paused until parse/schema/generation/title-label validity are >= 0.95 and JSON truncation <= 0.05.
- If raw_ends_after_json_rate >= 0.95, decoding is raw-clean. If raw_had_explanatory_tail_rate remains high but first-JSON extraction is stable, the output is parser-controlled rather than raw-clean.
- If raw tails remain high but first-JSON extraction is fully stable, Day2d can proceed only cautiously with the caveat: raw generation still contains explanatory tails; downstream analysis uses first JSON object extraction.
- If output control passes but matched-title hit rate remains 0.15-0.20, candidate-grounded generation is controllable but base Qwen recommendation choice is modest.
