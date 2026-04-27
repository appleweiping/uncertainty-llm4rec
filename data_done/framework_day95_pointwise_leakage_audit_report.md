# Framework-Day9.5 Pointwise Leakage / Evaluation Audit Report

## 1. Why Audit

The server-reported pointwise-v1 aggregated ranking was near oracle. That is suspicious for a 300-step small Qwen-LoRA baseline and must be audited before any success claim or Day10 scaling.

## 2. Prompt/Input Leakage Check

- status: `pass`
- output: `data_done/framework_day95_pointwise_prompt_leakage_check.csv`
- suspicious examples: `data_done/framework_day95_pointwise_suspicious_examples.jsonl`

This checks `input` and `metadata`; `output.relevance_label` is allowed because it is the supervised training target, not model input.

## 3. Split Overlap Check

- output: `data_done/framework_day95_pointwise_split_overlap_check.csv`
- note: `data_done_lora` pointwise samples do not preserve explicit `user_id`, so user_id+candidate overlap cannot be fully checked from this artifact alone.

## 4. Eval Logic Audit

- score source: parsed model `relevance_score` or parsed `relevance_label`
- parse failure fallback in original evaluator: score `0.0`
- original tie-breaker: original candidate order via `local_idx`
- concern: if positives are first in candidate order, ties/failures can inflate metrics

See `data_done/framework_day95_pointwise_eval_logic_audit.md`.

## 5. Candidate Order Bias

- test positive-at-position-1 rate: `1.0`
- output: `data_done/framework_day95_candidate_order_bias.csv`

If this rate is high, the original evaluator's order-preserving tie-break is unsafe.

## 6. Prediction Distribution

- prediction artifact present: `False`
- output: `data_done/framework_day95_pointwise_prediction_distribution.csv`

If prediction artifact is missing, sync `output-repaired/framework/day9_qwen_lora_beauty_pointwise_predictions.jsonl` from the server and rerun this audit.

## 7. Independent Safe Eval

- output: `data_done/framework_day95_pointwise_independent_eval.csv`
- rule: scores only from parsed model output; parse failure score `0.5`; tie-break by lexical candidate_item_id, not label or original order.

## 8. Conclusion

Current audit status: `missing_server_artifact`.

Do not write "pointwise baseline succeeded" until the independent safe eval is computed from the server prediction JSONL and reviewed.

## 9. Day10 Recommendation

First sync the missing Day9 server artifacts if needed, rerun:

```bash
python main_framework_day95_pointwise_leakage_audit.py
```

Only if independent safe eval remains strong and no leakage/order-bias issue is found should Day10 scale the selected baseline. Do not enter confidence/evidence/CEP framework yet.
