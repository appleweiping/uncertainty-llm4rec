# Framework-Day10 Server Commands

Run from the server repository:

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

If pull is blocked by server-generated summaries, back up the listed files under `server_local_backup/`, then `git restore` only the conflicting tracked summary files and pull again.

## 1. Build Shuffled Candidate-Order Data

```bash
python main_framework_day10_shuffle_lora_candidate_order.py --overwrite
```

Check:

```bash
cat data_done/framework_day10_candidate_order_diagnostics.csv
```

The old pointwise position-1 rate should be `1.0`; shuffled pointwise should be roughly uniform across positions 1-6.

## 2. Train + Eval Listwise Strict Shuffled

```bash
python main_framework_day10_train_qwen_lora_shuffled.py --name listwise_shuffled
python main_framework_day10_eval_qwen_lora_listwise_shuffled.py --eval_samples 512
```

Outputs:

```text
data_done/framework_day10_listwise_strict_shuffled_train_metrics.json
data_done/framework_day10_listwise_strict_shuffled_train_report.md
data_done/framework_day10_listwise_strict_shuffled_eval512_summary.csv
output-repaired/framework/day10_qwen_lora_beauty_listwise_strict_shuffled_eval512_predictions.jsonl
```

## 3. Optional Audited Pointwise Shuffled Comparison

Run only after listwise finishes:

```bash
python main_framework_day10_train_qwen_lora_shuffled.py --name pointwise_shuffled
python main_framework_day10_eval_qwen_lora_pointwise_shuffled.py --eval_users 512
```

Outputs:

```text
data_done/framework_day10_pointwise_shuffled_train_metrics.json
data_done/framework_day10_pointwise_shuffled_train_report.md
data_done/framework_day10_pointwise_shuffled_safe_eval_summary.csv
data_done/framework_day10_pointwise_shuffled_leakage_audit.md
output-repaired/framework/day10_qwen_lora_beauty_pointwise_shuffled_predictions.jsonl
```

## 4. Summarize

```bash
python main_framework_day10_summarize_candidate_order_repair.py
python scripts/framework_artifact_manifest.py
```

## 5. Sync Back To Local

```bash
tar -czf framework_light_reports_day10.tar.gz \
  data_done/framework_day10*.md \
  data_done/framework_day10*.csv \
  data_done/framework_day10*.json \
  data_done/framework_artifact_manifest_server.* \
  configs/framework \
  prompts/framework

tar -czf framework_day10_predictions_needed.tar.gz \
  output-repaired/framework/day10_qwen_lora_beauty_listwise_strict_shuffled_eval512_predictions.jsonl \
  output-repaired/framework/day10_qwen_lora_beauty_pointwise_shuffled_predictions.jsonl
```

Do not include `artifacts/lora/` unless adapter migration is explicitly needed.
