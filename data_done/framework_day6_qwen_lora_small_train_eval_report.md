# Framework-Day6 Qwen-LoRA Small Train + Eval Report

## 1. Day5 Tiny Train Recap

Day5 confirmed real server-side optimizer steps for Qwen3-8B + LoRA on Beauty listwise data: `loss_first=0.4933`, `loss_last=0.0090`, `loss_nan_count=0`, and peak GPU memory about `20.25 GB`.

## 2. Day6 Small Train Setup

Day6 trained the Beauty listwise baseline for `100` steps using `512` train samples, `batch_size=1`, `gradient_accumulation_steps=4`, `max_seq_len=2048`, LoRA rank `8`, alpha `16`, and dropout `0.05`.

## 3. Train Result

Training completed successfully with `loss_nan_count=0`, peak GPU memory `20.2553 GB`, and runtime `325.72 seconds`. The adapter was saved to `artifacts/lora/qwen3_8b_beauty_listwise_day6_small`, which is intentionally not committed.

## 4. Inference / Parser Result

On 128 Beauty test listwise samples, parse success was `0.8672` and schema valid rate was `0.8516`. Invalid item and duplicate item rates were both `0`, suggesting the main issue is output format/schema stability rather than candidate hallucination.

## 5. Ranking Metrics

The Day6 LoRA small adapter reached `NDCG@10=0.5683`, `MRR=0.4158`, `HR@1=0.2422`, `HR@3=0.4531`, `NDCG@3=0.3610`, and `NDCG@5=0.4823`. Random ranking on the same sample had `NDCG@10=0.5505` and `MRR=0.4076`, so the LoRA adapter is only slightly better on NDCG/MRR and weaker on some mid-rank metrics.

## 6. Baseline Comparison

This result should not be overclaimed. Day6 proves the pipeline works, not that the baseline is strong. The small adapter is not yet a stable recommender baseline.

## 7. Ready for Day7

Day7 should diagnose parse failures, inspect ranking success/failure cases, evaluate a larger subset without retraining, and decide whether to repair parser/prompt or train longer. Do not enter CEP/confidence/evidence framework until the local baseline is stable.

## 8. Limitations

This is small train and small eval only. HR@10 is trivial because Beauty 5neg has six candidates per user. No CEP framework, no confidence module, no evidence module, no pointwise training, and no four-domain training are involved.
