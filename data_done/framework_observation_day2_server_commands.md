# Framework-Observation-Day2 Server Commands

These commands run the Beauty 100-user candidate-grounded generative recommendation smoke. They do not train, do not run CEP, do not use evidence fields, and do not continue yes/no confidence.

## Server Run

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm

python main_framework_observation_day2_generative_recommendation_infer.py \
  --config configs/framework_observation/beauty_qwen_base_generative_candidate_grounded.yaml \
  --setting candidate_grounded \
  --model_variant base \
  --backend vllm \
  --split valid \
  --max_users 100 \
  --resume

python main_framework_observation_day2_generative_recommendation_infer.py \
  --config configs/framework_observation/beauty_qwen_base_generative_candidate_grounded.yaml \
  --setting candidate_grounded \
  --model_variant base \
  --backend vllm \
  --split test \
  --max_users 100 \
  --resume

python main_framework_observation_day2_generative_recommendation_analysis.py \
  --config configs/framework_observation/beauty_qwen_base_generative_candidate_grounded.yaml \
  --pred_dir output-repaired/framework_observation/beauty_qwen_base_generative_candidate_grounded/predictions
```

## Server Package

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
tar -czf framework_observation_day2_generative_results.tar.gz \
  data_done/framework_observation_day2_generative_uncertainty_literature_audit.md \
  data_done/framework_observation_day2_generative_task_definition.md \
  data_done/framework_observation_day2_original_research_direction_note.md \
  data_done/framework_observation_day2_generative_candidate_grounded_diagnostics.csv \
  data_done/framework_observation_day2_generative_candidate_grounded_calibration.csv \
  data_done/framework_observation_day2_generative_recommendation_report.md \
  data_done/framework_observation_day2_generative_vs_binary_observation_comparison.csv \
  data_done/framework_observation_day2_server_commands.md \
  output-repaired/framework_observation/beauty_qwen_base_generative_candidate_grounded/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_base_generative_candidate_grounded/predictions/test_raw.jsonl
```

## Local Pull From Windows

```powershell
scp ajifang@125.71.97.70:/home/ajifang/projects/uncertainty-llm4rec-week4/framework_observation_day2_generative_results.tar.gz .
```
