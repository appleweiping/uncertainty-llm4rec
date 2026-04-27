# Framework-Day8 Server Commands

Run these commands on the server after Day7 eval512 has produced raw predictions.

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

Parser-only repair and failure taxonomy on existing raw outputs:

```bash
python main_framework_day8_qwen_lora_output_repair.py
```

Strict prompt + deterministic generation + repaired parser re-eval on the existing Day6 adapter:

```bash
python main_framework_day8_qwen_lora_output_repair.py --run_strict_eval --eval_samples 512
```

If 512 is slow, use 256:

```bash
python main_framework_day8_qwen_lora_output_repair.py --run_strict_eval --eval_samples 256
```

Optional base Qwen comparison with strict prompt:

```bash
python main_framework_day8_qwen_lora_output_repair.py --run_base_comparison --base_eval_samples 128
```

Expected outputs:

```text
data_done/framework_day8_parse_failure_taxonomy.csv
data_done/framework_day8_parse_failure_examples_review.md
data_done/framework_day8_parser_repair_before_after.csv
data_done/framework_day8_generation_config_repair.md
prompts/framework/qwen_candidate_ranking_baseline_json_strict.txt
data_done/framework_day8_repaired_eval512_summary.csv
output-repaired/framework/day8_qwen_lora_beauty_listwise_eval512_repaired_predictions.jsonl
data_done/framework_day8_base_vs_lora_strict_prompt_comparison.csv
data_done/framework_day8_qwen_lora_output_repair_report.md
```

Do not train a new adapter. Do not run CEP/confidence/evidence framework. Do not commit adapter/checkpoint/artifact directories or large raw prediction JSONL.
