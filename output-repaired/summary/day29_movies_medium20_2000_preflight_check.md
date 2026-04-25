# Day29 Movies medium_20neg_2000 Preflight Check

- config: `configs\exp\movies_deepseek_relevance_evidence_medium_20neg_2000.yaml`
- train_input_path: `data\processed\amazon_movies_medium_20neg_2000\train.jsonl`
- valid_input_path: `data\processed\amazon_movies_medium_20neg_2000\valid.jsonl`
- test_input_path: `data\processed\amazon_movies_medium_20neg_2000\test.jsonl`
- prompt_path: `prompts/candidate_relevance_evidence.txt`
- output_dir: `output-repaired/movies_deepseek_relevance_evidence_medium_20neg_2000`
- output_schema: `relevance_evidence`
- resume: `True`
- concurrent: `True`

## Split Stats

| split | rows | users | pool_min | pool_mean | pool_max | positive | negative | missing_fields |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 158634 |  |  |  |  |  |  | not fully scanned |
| valid | 42000 | 2000 | 21 | 21.00 | 21 | 2000 | 40000 | candidate_title_optional:17063 |
| test | 42000 | 2000 | 21 | 21.00 | 21 | 2000 | 40000 | candidate_title_optional:17029 |

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