# Framework-Observation-Day2b Server Commands

These commands run the Beauty 100-user candidate-grounded title-generation prompt/parser repair smoke. This is not training, not CEP, not evidence, and not open-title generation.

## Server Run

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm

python main_framework_observation_day2_generative_recommendation_infer.py \
  --config configs/framework_observation/beauty_qwen_base_generative_candidate_grounded_day2b.yaml \
  --setting candidate_grounded \
  --model_variant base \
  --backend vllm \
  --split test \
  --max_users 100 \
  --resume

python main_framework_observation_day2_generative_recommendation_analysis.py \
  --config configs/framework_observation/beauty_qwen_base_generative_candidate_grounded_day2b.yaml \
  --pred_dir output-repaired/framework_observation/beauty_qwen_base_generative_candidate_grounded_day2b/predictions
```

Optional valid split for formal calibration:

```bash
python main_framework_observation_day2_generative_recommendation_infer.py \
  --config configs/framework_observation/beauty_qwen_base_generative_candidate_grounded_day2b.yaml \
  --setting candidate_grounded \
  --model_variant base \
  --backend vllm \
  --split valid \
  --max_users 100 \
  --resume

python main_framework_observation_day2_generative_recommendation_analysis.py \
  --config configs/framework_observation/beauty_qwen_base_generative_candidate_grounded_day2b.yaml \
  --pred_dir output-repaired/framework_observation/beauty_qwen_base_generative_candidate_grounded_day2b/predictions
```

## Server Package

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
tar -czf framework_observation_day2b_generative_repair_results.tar.gz \
  data_done/framework_observation_day2b_generative_candidate_grounded_repair_diagnostics.csv \
  data_done/framework_observation_day2b_generative_candidate_grounded_repair_calibration.csv \
  data_done/framework_observation_day2b_generative_candidate_grounded_repair_report.md \
  data_done/framework_observation_day2b_day2_vs_day2b_comparison.csv \
  data_done/framework_observation_day2c_label_first_generation_plan.md \
  data_done/framework_observation_day2b_server_commands.md \
  output-repaired/framework_observation/beauty_qwen_base_generative_candidate_grounded_day2b/predictions/test_raw.jsonl
```

## Local Pull From Windows

```powershell
scp ajifang@125.71.97.70:/home/ajifang/projects/uncertainty-llm4rec-week4/framework_observation_day2b_generative_repair_results.tar.gz .
```
