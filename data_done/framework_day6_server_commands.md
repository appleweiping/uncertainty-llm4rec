# Framework-Day6 Server Commands

Run these commands on the server.

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

Confirm data is materialized:

```bash
test -f data_done_lora/beauty/train_listwise.jsonl
test -f data_done_lora/beauty/valid_listwise.jsonl
test -f data_done_lora/beauty/test_listwise.jsonl
python scripts/check_framework_server_readiness.py
```

Train the Day6 Beauty listwise small LoRA adapter:

```bash
python main_framework_day6_train_qwen_lora_small.py --config configs/framework/qwen3_8b_lora_baseline_beauty_listwise_small.yaml
```

Evaluate closed-catalog ranking on a small Beauty test subset:

```bash
python main_framework_day6_eval_qwen_lora_listwise.py --config configs/framework/qwen3_8b_lora_baseline_beauty_listwise_small.yaml --adapter_path artifacts/lora/qwen3_8b_beauty_listwise_day6_small --max_eval_samples 128
```

View reports:

```bash
cat data_done/framework_day6_beauty_listwise_small_train_report.md
cat data_done/framework_day6_beauty_listwise_eval_report.md
cat data_done/framework_day6_qwen_lora_small_train_eval_report.md
```

Expected outputs:

```text
data_done/framework_day6_beauty_listwise_small_train_metrics.json
data_done/framework_day6_beauty_listwise_small_train_report.md
output-repaired/framework/day6_qwen_lora_beauty_listwise_predictions.jsonl
data_done/framework_day6_beauty_listwise_eval_summary.csv
data_done/framework_day6_beauty_listwise_baseline_comparison.csv
data_done/framework_day6_beauty_listwise_eval_report.md
data_done/framework_day6_qwen_lora_small_train_eval_report.md
artifacts/lora/qwen3_8b_beauty_listwise_day6_small/
```

Do not commit `artifacts/`, prediction JSONL, checkpoints, or raw logs. HR@10 is trivial because Beauty 5neg has six candidates per user; use NDCG@10, MRR, HR@1, HR@3, NDCG@3, and NDCG@5 as primary metrics.
