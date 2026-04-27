# Framework-Observation-Day1f Server Commands

Day1f is a Beauty-only self-consistency confidence audit. It does not train, use evidence, use CEP, call external APIs, or run four domains.

## Server Run

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm

python main_framework_observation_day1f_self_consistency.py --config configs/framework_observation/beauty_qwen_lora_self_consistency.yaml --create_subset

python main_framework_observation_day1f_self_consistency.py --config configs/framework_observation/beauty_qwen_lora_self_consistency.yaml --run_self_consistency valid --model_variant lora --resume
python main_framework_observation_day1f_self_consistency.py --config configs/framework_observation/beauty_qwen_lora_self_consistency.yaml --run_self_consistency test --model_variant lora --resume

python main_framework_observation_day1f_self_consistency.py --config configs/framework_observation/beauty_qwen_lora_self_consistency.yaml --run_logit_subset valid --model_variant lora --resume
python main_framework_observation_day1f_self_consistency.py --config configs/framework_observation/beauty_qwen_lora_self_consistency.yaml --run_logit_subset test --model_variant lora --resume

python main_framework_observation_day1f_self_consistency.py --config configs/framework_observation/beauty_qwen_lora_self_consistency.yaml --analyze_only
```

Run each command sequentially. Do not launch valid/test concurrently because both vLLM and the transformers logit scorer need the same GPU.

If a stale vLLM process keeps GPU memory after an interrupted run:

```bash
nvidia-smi
ps -u ajifang -f | grep -E "python|vllm|EngineCore" | grep -v grep
kill <VLLM_ENGINECORE_PID>
```

## Server Package

```bash
python scripts/framework_artifact_manifest.py
tar -czf framework_observation_day1f_self_consistency_results.tar.gz \
  data_done/framework_observation_day1f_self_consistency_subset.json \
  data_done/framework_observation_day1f_self_consistency_diagnostics.csv \
  data_done/framework_observation_day1f_self_consistency_calibration.csv \
  data_done/framework_observation_day1f_self_consistency_ranking_eval.csv \
  data_done/framework_observation_day1f_self_consistency_report.md \
  data_done/framework_observation_day1f_logit_vs_self_consistency_comparison.csv \
  data_done/framework_observation_day1f_go_no_go_decision.md \
  data_done/framework_observation_day1g_pair_list_context_plan.md \
  data_done/framework_artifact_manifest_server.json \
  data_done/framework_artifact_manifest_server.md \
  output-repaired/framework_observation/beauty_qwen_lora_self_consistency/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_self_consistency/predictions/test_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_logit_confidence_day1f_subset/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_logit_confidence_day1f_subset/predictions/test_raw.jsonl
```

## Local Pull

```powershell
cd D:\Research\Uncertainty-LLM4Rec
scp ajifang@125.71.97.70:/home/ajifang/projects/uncertainty-llm4rec-week4/framework_observation_day1f_self_consistency_results.tar.gz .
tar -xzf framework_observation_day1f_self_consistency_results.tar.gz
```
