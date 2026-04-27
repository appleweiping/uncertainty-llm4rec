# Framework Server Artifact Sync Commands

Run these on the server from:

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
```

## 1. Generate Manifest

```bash
python scripts/framework_artifact_manifest.py
```

## 2. Pack Lightweight Framework Reports

```bash
tar -czf framework_light_reports_day9.tar.gz \
  data_done/framework_day9*.md \
  data_done/framework_day9*.csv \
  data_done/framework_day9*.json \
  data_done/framework_day95*.md \
  data_done/framework_day95*.csv \
  data_done/framework_day95*.json \
  data_done/framework_artifact_manifest_server.* \
  output-repaired/framework/*summary* \
  output-repaired/framework/*eval*summary* \
  configs/framework \
  prompts/framework
```

Use this bundle for local/Codex report review. It should not include adapter weights.

## 3. Pack Prediction JSONL Only When Needed

For Day9.5 leakage audit, local Codex needs the pointwise prediction JSONL:

```bash
tar -czf framework_day9_predictions_needed.tar.gz \
  output-repaired/framework/day9_qwen_lora_beauty_pointwise_predictions.jsonl \
  output-repaired/framework/day9_qwen_lora_beauty_listwise_strict_eval512_predictions.jsonl
```

If the listwise strict prediction filename differs, inspect:

```bash
ls output-repaired/framework/*day9*qwen*lora*jsonl
```

## 4. Do Not Pack Adapters By Default

Do not include:

```text
artifacts/lora/
artifacts/lora_smoke/
```

Only package adapter directories if we explicitly decide to migrate adapter weights between machines.
