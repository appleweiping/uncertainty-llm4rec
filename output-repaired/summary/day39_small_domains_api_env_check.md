# Day39 Small Domains API Environment Check

Day39 inherits the Day37 movies_small successful route. The earlier Day37 APIConnectionError was caused by proxy variables pointing to a bad local proxy; Day39 commands should clear those proxy variables before calling `main_infer.py`.

## Proxy Environment

- `HTTP_PROXY` present: `True`
- `HTTPS_PROXY` present: `True`
- `ALL_PROXY` present: `True`
- `http_proxy` present: `True`
- `https_proxy` present: `True`
- `all_proxy` present: `True`

## DeepSeek Route

- backend/provider: `deepseek` / `deepseek`
- model_name: `deepseek-chat`
- base_url: `https://api.deepseek.com`
- api_key_env: `DEEPSEEK_API_KEY` (value not printed)
- api_key_env_present: `True`
- prompt_path: `prompts/candidate_relevance_evidence.txt`
- output_schema: `relevance_evidence`
- stable command shape: `py -3.12 main_infer.py --config <domain_config> --split_name valid/test --concurrent --resume --max_workers 4 --requests_per_minute 120`

Reference: Day37 movies_small completed with this route after clearing proxy variables.
