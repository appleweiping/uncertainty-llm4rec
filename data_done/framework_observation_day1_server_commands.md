# Framework-Observation-Day1 Server Commands

Scope: Beauty-only local Qwen-LoRA raw confidence observation. This does not call external APIs, does not use evidence fields, and does not implement CEP fusion.

Run from the server repository:

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

## 1. LoRA Confidence Smoke

Use the configured LoRA adapter:

```text
artifacts/lora/qwen3_8b_beauty_listwise_strict_day9_small
```

If this adapter is not present, update `configs/framework_observation/beauty_qwen_lora_confidence.yaml` to the intended single adapter path before running. Do not mix multiple adapters in one result directory.

Run 200-row smoke on valid/test:

```bash
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence.yaml --split valid --model_variant lora --max_samples 200 --resume
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence.yaml --split test --model_variant lora --max_samples 200 --resume
python main_framework_observation_day1_confidence_analysis.py --pred_dir output-repaired/framework_observation/beauty_qwen_lora_confidence/predictions
```

Check:

```bash
cat data_done/framework_observation_day1_beauty_confidence_diagnostics.csv
```

Only continue if parse/schema quality is acceptable, ideally `schema_valid_rate >= 0.9`.

## 2. LoRA Confidence Full Beauty 5neg

Resume into the same output files:

```bash
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence.yaml --split valid --model_variant lora --resume
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence.yaml --split test --model_variant lora --resume
python main_framework_observation_day1_confidence_analysis.py --pred_dir output-repaired/framework_observation/beauty_qwen_lora_confidence/predictions
python scripts/framework_artifact_manifest.py
```

Expected prediction outputs:

```text
output-repaired/framework_observation/beauty_qwen_lora_confidence/predictions/valid_raw.jsonl
output-repaired/framework_observation/beauty_qwen_lora_confidence/predictions/test_raw.jsonl
```

## 3. Optional Base Qwen Control

Do not run by default. If needed later:

```bash
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_base_confidence.yaml --split valid --model_variant base --max_samples 200 --resume
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_base_confidence.yaml --split test --model_variant base --max_samples 200 --resume
```

## 4. Sync Back To Local

Light reports:

```bash
tar -czf framework_observation_day1_light_reports.tar.gz \
  data_done/framework_observation_day1*.md \
  data_done/framework_observation_day1*.csv \
  data_done/framework_artifact_manifest_server.* \
  configs/framework_observation \
  prompts/framework/qwen_confidence_recommendation_pointwise.txt
```

Prediction JSONL, only when local audit needs it:

```bash
tar -czf framework_observation_day1_predictions_needed.tar.gz \
  output-repaired/framework_observation/beauty_qwen_lora_confidence/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_confidence/predictions/test_raw.jsonl
```

Do not package adapter weights unless explicitly needed.
