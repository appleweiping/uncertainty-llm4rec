# Framework-Day8 Generation Config Repair

The Day8 inference path uses deterministic generation for closed-catalog JSON ranking.

## Repair

- `do_sample = false`
- Do not pass `temperature`, `top_p`, or `top_k` when `do_sample=false`; Transformers may warn that these flags are ignored.
- `max_new_tokens = 128` by default, enough for six Beauty 5neg candidate IDs. Use `192` only if outputs are truncated.
- Set `pad_token_id` from tokenizer pad token, falling back to EOS when needed.
- Set `eos_token_id` from tokenizer EOS.

## Boundary

This is an inference-time generation cleanup only. It does not change training data, does not train a new adapter, and does not add confidence/evidence/CEP fields.
