# Framework-Day9 Server Commands

Run from the server repository:

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

If `git pull` is blocked by locally edited result files, back up those files under `server_local_backup/`, restore only the conflicting tracked summaries, then pull again. Do not delete adapters or prediction artifacts.

## 1. Materialize Strict Listwise Data

```bash
python main_framework_day9_build_strict_listwise_data.py --domain beauty --input_root data_done --output_root data_done_lora --overwrite
```

Expected generated files:

```text
data_done_lora/beauty/train_listwise_json_strict.jsonl
data_done_lora/beauty/valid_listwise_json_strict.jsonl
data_done_lora/beauty/test_listwise_json_strict.jsonl
data_done/framework_day9_listwise_strict_data_stats.json
```

The JSONL files are local data artifacts and should not be committed.

## 2. Train Listwise-v2 Strict Baseline

```bash
python main_framework_day9_train_qwen_lora_formulation.py --name listwise_strict --config configs/framework/qwen3_8b_lora_baseline_beauty_listwise_strict_small.yaml
```

Then evaluate:

```bash
python main_framework_day9_eval_qwen_lora_listwise_strict.py --config configs/framework/qwen3_8b_lora_baseline_beauty_listwise_strict_small.yaml --adapter_path artifacts/lora/qwen3_8b_beauty_listwise_strict_day9_small --eval_samples 512
```

## 3. Train Pointwise-v1 Baseline

```bash
python main_framework_day9_train_qwen_lora_formulation.py --name pointwise --config configs/framework/qwen3_8b_lora_baseline_beauty_pointwise_small.yaml
```

Then evaluate. This runs pointwise inference for 512 users, i.e. 3072 candidate calls under the 5neg pool:

```bash
python main_framework_day9_eval_qwen_lora_pointwise.py --config configs/framework/qwen3_8b_lora_baseline_beauty_pointwise_small.yaml --adapter_path artifacts/lora/qwen3_8b_beauty_pointwise_day9_small --eval_users 512
```

If runtime is too high, use `--eval_users 128` first, then rerun 512 after sanity checks pass.

## 4. Summarize Baselines

```bash
python main_framework_day9_summarize_baseline_formulations.py
```

Outputs:

```text
data_done/framework_day9_listwise_strict_train_metrics.json
data_done/framework_day9_listwise_strict_eval512_summary.csv
data_done/framework_day9_pointwise_train_metrics.json
data_done/framework_day9_pointwise_eval_summary.csv
data_done/framework_day9_qwen_lora_baseline_formulation_comparison.csv
data_done/framework_day9_qwen_lora_baseline_formulation_report.md
```

Do not commit adapters under `artifacts/` or raw prediction JSONL under `output-repaired/framework/`.
