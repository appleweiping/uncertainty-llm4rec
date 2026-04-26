# Day37 Movies Small API Parity Check

## Config Parity

`movies_small` uses the same successful route as Day29 Movies medium5 and Day30 robustness:

- backend/provider: `deepseek` / `deepseek`
- model_name: `deepseek-chat`
- base_url: `https://api.deepseek.com`
- api_key_env: `DEEPSEEK_API_KEY` (value not printed)
- prompt_path: `prompts/candidate_relevance_evidence.txt`
- output_schema: `relevance_evidence`
- main command: `py -3.12 main_infer.py --config configs\exp\movies_small_deepseek_relevance_evidence.yaml --split_name valid/test --concurrent --resume --max_workers 4 --requests_per_minute 120`
- working directory: `D:\Research\Uncertainty-LLM4Rec`

## Root Cause Of Earlier APIConnectionError

The Day37 config matched the successful Day29/Day30 config. The failure was caused by shell proxy environment variables pointing to a bad local proxy (`127.0.0.1:9`). Clearing `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` restored the same `main_infer.py` route.

Current process proxy env present flags: `{'HTTP_PROXY': True, 'HTTPS_PROXY': True, 'ALL_PROXY': True}`.

## Recovery Result

After clearing proxy variables for the inference command:

- one-row parity health succeeded;
- 20-row movies_small smoke succeeded;
- movies_small valid/test full inference completed with parse_success=1.0.

No prompt, parser, formula, backend, or model config changes were made.
