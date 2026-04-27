# Framework-Observation-Day1i Server Commands

Day1i is a Beauty-only candidate-order shuffled control for listwise behavioral uncertainty. It does not train, use evidence, implement CEP, call external APIs, or run four domains.

## Server Run

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm

python main_framework_observation_day1i_shuffled_behavioral_uncertainty.py --config configs/framework_observation/beauty_qwen_lora_day1i_shuffled_behavioral_uncertainty.yaml --create_shuffle_subset

python main_framework_observation_day1i_shuffled_behavioral_uncertainty.py --config configs/framework_observation/beauty_qwen_lora_day1i_shuffled_behavioral_uncertainty.yaml --run_inference valid --model_variant lora --resume
python main_framework_observation_day1i_shuffled_behavioral_uncertainty.py --config configs/framework_observation/beauty_qwen_lora_day1i_shuffled_behavioral_uncertainty.yaml --run_inference test --model_variant lora --resume

python main_framework_observation_day1i_shuffled_behavioral_uncertainty.py --config configs/framework_observation/beauty_qwen_lora_day1i_shuffled_behavioral_uncertainty.yaml --analyze_only
```

Run commands sequentially. Do not run valid/test concurrently on the same GPU.

## Server Package

```bash
python scripts/framework_artifact_manifest.py
tar -czf framework_observation_day1i_shuffled_behavioral_uncertainty_results.tar.gz \
  data_done/framework_observation_day1i_shuffled_candidate_order_subset.json \
  data_done/framework_observation_day1i_candidate_order_diagnostics.csv \
  data_done/framework_observation_day1i_shuffled_behavioral_ranking_eval.csv \
  data_done/framework_observation_day1i_shuffled_behavioral_uncertainty_calibration.csv \
  data_done/framework_observation_day1i_shuffled_behavioral_diagnostics.csv \
  data_done/framework_observation_day1i_order_bias_control_comparison.csv \
  data_done/framework_observation_day1i_order_shuffled_behavioral_uncertainty_report.md \
  data_done/framework_observation_day1i_go_no_go_decision.md \
  data_done/framework_artifact_manifest_server.json \
  data_done/framework_artifact_manifest_server.md \
  output-repaired/framework_observation/beauty_qwen_lora_listwise_behavioral_shuffled/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_listwise_behavioral_shuffled/predictions/test_raw.jsonl
```

## Local Pull

```powershell
cd D:\Research\Uncertainty-LLM4Rec
scp ajifang@125.71.97.70:/home/ajifang/projects/uncertainty-llm4rec-week4/framework_observation_day1i_shuffled_behavioral_uncertainty_results.tar.gz .
tar -xzf framework_observation_day1i_shuffled_behavioral_uncertainty_results.tar.gz
```
