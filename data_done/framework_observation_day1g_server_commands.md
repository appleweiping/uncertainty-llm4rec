# Framework-Observation-Day1g Server Commands

Day1g is a Beauty-only relative context audit. It does not train, use evidence, implement CEP, call external APIs, or run four domains.

## Server Run

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm

python main_framework_observation_day1g_list_pair_context_audit.py --config configs/framework_observation/beauty_qwen_lora_day1g_context.yaml --run_listwise valid --model_variant lora --resume
python main_framework_observation_day1g_list_pair_context_audit.py --config configs/framework_observation/beauty_qwen_lora_day1g_context.yaml --run_listwise test --model_variant lora --resume

python main_framework_observation_day1g_list_pair_context_audit.py --config configs/framework_observation/beauty_qwen_lora_day1g_context.yaml --run_pairwise valid --model_variant lora --resume
python main_framework_observation_day1g_list_pair_context_audit.py --config configs/framework_observation/beauty_qwen_lora_day1g_context.yaml --run_pairwise test --model_variant lora --resume

python main_framework_observation_day1g_list_pair_context_audit.py --config configs/framework_observation/beauty_qwen_lora_day1g_context.yaml --analyze_only
```

Run commands sequentially. Do not run valid/test or listwise/pairwise in parallel on the same GPU.

If pairwise runtime is too high, listwise can be analyzed first after the two listwise commands. Pairwise outputs will be marked pending until pairwise predictions exist.

## Server Package

```bash
python scripts/framework_artifact_manifest.py
tar -czf framework_observation_day1g_context_audit_results.tar.gz \
  data_done/framework_observation_day1g_listwise_context_diagnostics.csv \
  data_done/framework_observation_day1g_listwise_context_calibration.csv \
  data_done/framework_observation_day1g_listwise_context_report.md \
  data_done/framework_observation_day1g_pairwise_context_diagnostics.csv \
  data_done/framework_observation_day1g_pairwise_context_calibration.csv \
  data_done/framework_observation_day1g_pairwise_context_ranking_eval.csv \
  data_done/framework_observation_day1g_context_comparison.csv \
  data_done/framework_observation_day1g_go_no_go_decision.md \
  data_done/framework_artifact_manifest_server.json \
  data_done/framework_artifact_manifest_server.md \
  output-repaired/framework_observation/beauty_qwen_lora_listwise_context/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_listwise_context/predictions/test_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_pairwise_context/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_pairwise_context/predictions/test_raw.jsonl
```

## Local Pull

```powershell
cd D:\Research\Uncertainty-LLM4Rec
scp ajifang@125.71.97.70:/home/ajifang/projects/uncertainty-llm4rec-week4/framework_observation_day1g_context_audit_results.tar.gz .
tar -xzf framework_observation_day1g_context_audit_results.tar.gz
```
