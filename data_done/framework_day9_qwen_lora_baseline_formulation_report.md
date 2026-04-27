# Framework-Day9 Qwen-LoRA Baseline Formulation Report

## 1. Day8 Recap

Day8 showed that parser-only repair was marginal and strict inference prompt hurt the existing adapter because of prompt mismatch. The baseline problem is therefore formulation stability, not just parsing.

## 2. Why Formulation Repair

The CEP/confidence/evidence framework should not be layered on top of an unstable local recommender. Day9 compares listwise-v1, listwise-v2 strict train/infer, and pointwise-v1 aggregation before any framework fusion.

## 3. Listwise-v2 Strict Train/Infer

- status: `pending_server_run`
- prompt style: `json_strict_train_and_infer`
- target: closed-catalog full candidate ranking JSON

If this row remains `pending_server_run`, run the Day9 server commands after pulling the branch.

## 4. Pointwise-v1 Train/Aggregate

- status: `pending_server_run`
- formulation: candidate relevance label per item, aggregated into per-user ranking
- note: this is a raw relevance baseline, not calibrated probability and not CEP.

Important Day9.5 boundary: the server-reported pointwise-v1 result is suspiciously near-oracle. Because the local workspace does not yet contain the server prediction JSONL or pointwise eval summary, this report must not treat pointwise-v1 as a successful baseline. Use `data_done/framework_day95_pointwise_leakage_audit_report.md` after syncing server artifacts.

## 5. Comparison With Random and Day6

See `data_done/framework_day9_qwen_lora_baseline_formulation_comparison.csv`. HR@10 remains trivial for Beauty 5neg; use NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5.

## 6. Baseline Choice

Choose the next baseline only after server rows are filled. If pointwise-v1 is more stable, use it for larger Beauty training. If listwise-v2 is stronger, keep listwise. If both are weak, revise training target/data volume before entering CEP.

## 7. Day10 Recommendation

Do not enter confidence/evidence/CEP framework yet unless a baseline formulation clearly beats random with good parse/schema stability. Day10 should either scale the better formulation or revise the baseline prompt/data target.

## 8. Boundary

Day9 is still baseline-only. It does not call APIs, train four domains, use calibrated confidence, use evidence risk, or implement CEP fusion.
