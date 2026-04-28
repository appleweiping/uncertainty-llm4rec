# Framework-Observation-Day2c-Repair Server Commands

These commands run the Beauty 100-user label-first compact title-generation repair. This is not training, not evidence, not CEP, not external API use, not open-title generation, and not a full run.

## Server Run

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm

python main_framework_observation_day2c_label_first_generation.py \
  --config configs/framework_observation/beauty_qwen_base_generative_label_first_repair.yaml \
  --run_inference valid \
  --model_variant base \
  --backend vllm \
  --max_users 100 \
  --resume

python main_framework_observation_day2c_label_first_generation.py \
  --config configs/framework_observation/beauty_qwen_base_generative_label_first_repair.yaml \
  --run_inference test \
  --model_variant base \
  --backend vllm \
  --max_users 100 \
  --resume

python main_framework_observation_day2c_label_first_generation.py \
  --config configs/framework_observation/beauty_qwen_base_generative_label_first_repair.yaml \
  --analyze_only
```

## Server Package

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
tar -czf framework_observation_day2c_repair_label_first_results.tar.gz \
  data_done/framework_observation_day2c_repair_label_first_generation_diagnostics.csv \
  data_done/framework_observation_day2c_repair_label_first_generation_report.md \
  data_done/framework_observation_day2c_repair_control_comparison.csv \
  data_done/framework_observation_day2c_label_only_fallback_plan.md \
  data_done/framework_observation_day2c_repair_server_commands.md \
  output-repaired/framework_observation/beauty_qwen_base_generative_label_first_repair/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_base_generative_label_first_repair/predictions/test_raw.jsonl
```

## Local Pull From Windows

```powershell
scp ajifang@125.71.97.70:/home/ajifang/projects/uncertainty-llm4rec-week4/framework_observation_day2c_repair_label_first_results.tar.gz .
```
