# Framework-Day9 Generation Config Fix

Day8 server logs still showed:

```text
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k'].
```

Day9 fixes this at the inference entrypoints. For deterministic evaluation, we keep:

- `do_sample=false`
- `max_new_tokens`
- `pad_token_id`
- `eos_token_id`

We do not pass `temperature`, `top_p`, or `top_k` as generation kwargs. The shared listwise evaluation code also sanitizes `model.generation_config` by setting deterministic mode and clearing sampling-only fields before inference. This is an evaluation reproducibility fix only; it does not change the task definition, prompt schema, training labels, or CEP/framework logic.

Affected code:

- `main_framework_day6_eval_qwen_lora_listwise.py`
- `main_framework_day7_qwen_lora_eval_diagnosis.py`
- `main_framework_day9_eval_qwen_lora_listwise_strict.py`
- `main_framework_day9_eval_qwen_lora_pointwise.py`

Boundary: Day9 still does not implement confidence, evidence, calibrated posterior, or CEP fusion.
