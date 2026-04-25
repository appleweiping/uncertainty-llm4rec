# Day29 Movies medium_5neg_2000 Preflight Check

## 20neg_2000 Pause Note

Movies medium_20neg_2000 preflight and tiny smoke passed, and partial valid inference is preserved. Full 20neg_2000 inference was intentionally paused due to API/runtime cost.

- config: `configs\exp\movies_deepseek_relevance_evidence_medium_5neg_2000.yaml`
- train_input_path: `data\processed\amazon_movies_medium_5neg\train.jsonl`
- valid_input_path: `data\processed\amazon_movies_medium_5neg\valid.jsonl`
- test_input_path: `data\processed\amazon_movies_medium_5neg\test.jsonl`
- prompt_path: `prompts/candidate_relevance_evidence.txt`
- output_dir: `output-repaired/movies_deepseek_relevance_evidence_medium_5neg_2000`
- output_schema: `relevance_evidence`
- resume: `True`
- concurrent: `True`

## Split Stats

| split | rows | users | pool_min | pool_mean | pool_max | positive | negative | missing_fields |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 45324 |  |  |  |  |  |  | not fully scanned |
| valid | 12000 | 2000 | 6 | 6.00 | 6 | 2000 | 10000 | candidate_title_optional:5074 |
| test | 12000 | 2000 | 6 | 6.00 | 6 | 2000 | 10000 | candidate_title_optional:5116 |

## Metric Note

Each user has 6 candidates, so `hr10_trivial_flag=true`. HR@10 must not be used as primary evidence; use NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.

## Checks

- config_exists: `True`
- prompt_path_ok: `True`
- output_dir_ok: `True`
- schema_ok: `True`
- resume_enabled: `True`
- valid_rows_ok: `True`
- test_rows_ok: `True`
- valid_candidate_pool_ok: `True`
- test_candidate_pool_ok: `True`
- valid_schema_ok: `True`
- test_schema_ok: `True`
- output_dir_isolated: `True`

## Decision

Preflight passed. Tiny API smoke can start.