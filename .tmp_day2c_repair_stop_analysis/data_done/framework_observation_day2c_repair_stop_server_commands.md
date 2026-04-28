# Framework-Observation-Day2c-Repair-Stop Commands

Purpose: run the Beauty 100-user valid/test label-first candidate-grounded generation output-control repair. This keeps the Day2c-repair compact schema and `max_new_tokens=160`, adds vLLM stop variants, and analyzes with first-JSON extraction while separately reporting raw explanatory tails.

This is still observation only: no training, no evidence, no CEP, no open-title generation, no full run, and no external API.

## Server Run

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm

python main_framework_observation_day2c_label_first_generation.py \
  --config configs/framework_observation/beauty_qwen_base_generative_label_first_repair_stop.yaml \
  --run_inference valid \
  --model_variant base \
  --resume

python main_framework_observation_day2c_label_first_generation.py \
  --config configs/framework_observation/beauty_qwen_base_generative_label_first_repair_stop.yaml \
  --run_inference test \
  --model_variant base \
  --resume

python main_framework_observation_day2c_label_first_generation.py \
  --config configs/framework_observation/beauty_qwen_base_generative_label_first_repair_stop.yaml \
  --analyze_only
```

## Server Package

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
tar -czf framework_observation_day2c_repair_stop_results.tar.gz \
  data_done/framework_observation_day2c_repair_stop_diagnostics.csv \
  data_done/framework_observation_day2c_repair_stop_report.md \
  data_done/framework_observation_day2c_repair_stop_control_comparison.csv \
  data_done/framework_observation_day2c_repair_stop_server_commands.md \
  output-repaired/framework_observation/beauty_qwen_base_generative_label_first_repair_stop/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_base_generative_label_first_repair_stop/predictions/test_raw.jsonl
```

## Local Fetch

Run this from Windows PowerShell:

```powershell
scp ajifang@125.71.97.70:/home/ajifang/projects/uncertainty-llm4rec-week4/framework_observation_day2c_repair_stop_results.tar.gz D:\Research\Uncertainty-LLM4Rec\
```

## Expected Decision Logic

- If `parse_success_rate`, `schema_valid_rate`, `generation_valid_rate`, and `title_matches_selected_label_rate` are all at least `0.95`, and `json_truncation_rate <= 0.05`, output control is evaluable.
- If `raw_ends_after_json_rate >= 0.95`, decoding is raw-clean.
- If `raw_had_explanatory_tail_rate` remains high but first-JSON extraction is stable, this is parser-controlled output, not raw-clean output.
- Day2d non-verbal uncertainty should only start cautiously after output control is evaluable.
