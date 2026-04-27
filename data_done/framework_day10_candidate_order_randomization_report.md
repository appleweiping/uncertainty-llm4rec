# Framework-Day10 Candidate Order Randomization Report

## 1. Day9.5 Recap

Day9.5 showed that the pointwise near-oracle result was an order-bias artifact: old pointwise positives were fixed at position 1, and the original evaluator used original candidate order as the tie-break.

## 2. Why Candidate-Order Randomization Is Necessary

Candidate order must be neutral before training or evaluating Qwen-LoRA baselines. Otherwise parse failures, tied scores, or label-like ordering can create inflated ranking metrics.

## 3. Shuffled Data Construction

Day10 writes Beauty-only shuffled instruction data under `data_done_lora_v2/beauty/` and does not overwrite old `data_done_lora/beauty/`.

## 4. Positive Position Diagnostics

See `data_done/framework_day10_candidate_order_diagnostics.csv`. Old pointwise has positive position-1 rate `1.0`; shuffled pointwise spreads positives across positions 1-6.

## 5. Listwise Strict Shuffled Train/Eval

- status / NDCG@10: `pending_server_run`
- tie-break: `lexical`

This is the primary Day10 candidate baseline path.

## 6. Pointwise Shuffled Audited Eval

- status / NDCG@10: `pending_server_run`
- role: audited comparison only, not the main baseline unless safe eval clearly beats random.

## 7. Comparison

See `data_done/framework_day10_candidate_order_repaired_baseline_comparison.csv`.

## 8. Decision

If listwise strict shuffled is stable and beats random under safe eval, use it as the Day11 baseline candidate. If both listwise and pointwise remain weak, Day11 should redesign the baseline target rather than enter CEP.

## 9. Boundary

Day10 does not call APIs, train four domains, or implement confidence/evidence/CEP framework.
