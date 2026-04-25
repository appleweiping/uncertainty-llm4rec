# Day10-Full Readiness Check

Status: ready_for_smoke_or_full

Day10-full keeps the Day10 design: same Beauty users, same candidate pools, same DeepSeek list backend, same top-K and evaluation; the only methodological difference is whether scheme-four evidence fields are present.

## Checks

- PASS: same_input_path -- data/processed/amazon_beauty/ranking_test.jsonl
- PASS: same_user_count -- 973
- PASS: same_user_id_order -- 973
- PASS: same_candidate_pool_hash -- ea3fca636ec44ec4
- PASS: same_top_k -- 10
- PASS: same_model_config -- configs/model/deepseek_list.yaml
- PASS: plain_prompt -- prompts/recommendation_list_plain.txt
- PASS: evidence_prompt -- prompts/recommendation_list_evidence.txt
- PASS: parser_reuses_day10 -- parse_recommendation_list_plain_response / parse_recommendation_list_evidence_response
- PASS: resume_enabled -- resume=true
- PASS: concurrent_api_controlled -- workers=4 rpm=120
- PASS: separate_output_dirs -- output-repaired\beauty_deepseek_recommendation_list_full_plain vs output-repaired\beauty_deepseek_recommendation_list_full_evidence
- PASS: does_not_cover_day10_200 -- separate full exp_names
- PASS: full_sample_count -- 973
- PASS: candidate_pool_size_is_6 -- {6: 973}
- PASS: plain_smoke_or_full_status -- {'exists': True, 'rows': 973, 'parse_success_rate': 0.998972250770812, 'schema_valid_rate': 0.998972250770812}
- PASS: evidence_smoke_or_full_status -- {'exists': True, 'rows': 973, 'parse_success_rate': 0.998972250770812, 'schema_valid_rate': 0.998972250770812}

## Smoke Commands

Run these before full inference:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; & 'C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe' main_infer_recommendation_list.py --config configs/exp/beauty_deepseek_recommendation_list_full_plain.yaml --setting plain --max_samples 20
$env:PYTHONDONTWRITEBYTECODE='1'; & 'C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe' main_infer_recommendation_list.py --config configs/exp/beauty_deepseek_recommendation_list_full_evidence.yaml --setting evidence --max_samples 20
$env:PYTHONDONTWRITEBYTECODE='1'; & 'C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe' main_day10_full_readiness.py
```

Before full inference, require both smoke outputs to have parse_success_rate=1.0 and schema_valid_rate=1.0. After full inference, a parse_success_rate >=0.99 is treated as healthy and reported explicitly.
