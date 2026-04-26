# Day37 Movies Small Runtime Monitor

| split | expected_rows | current_rows | parse_success | status | resume_command |
|---|---:|---:|---:|---|---|
| valid | 3000 | 3000 | 1.0000 | complete | `$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''; py -3.12 main_infer.py --config configs\exp\movies_small_deepseek_relevance_evidence.yaml --split_name valid --concurrent --resume --max_workers 4 --requests_per_minute 120` |
| test | 3000 | 3000 | 1.0000 | complete | `$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''; py -3.12 main_infer.py --config configs\exp\movies_small_deepseek_relevance_evidence.yaml --split_name test --concurrent --resume --max_workers 4 --requests_per_minute 120` |
