# Framework-Day7 Server Commands

Run these commands on the server after Day6 has produced the adapter and prediction JSONL.

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

First run parser / case-study diagnosis on the existing Day6 predictions:

```bash
python main_framework_day7_qwen_lora_eval_diagnosis.py
```

Run larger eval with the Day6 adapter, without retraining:

```bash
python main_framework_day7_qwen_lora_eval_diagnosis.py --run_eval512 --eval_samples 512
```

If 512 is too slow, use 256:

```bash
python main_framework_day7_qwen_lora_eval_diagnosis.py --run_eval512 --eval_samples 256
```

Optional base Qwen comparison on the same small subset. This can be slower, so run only if needed:

```bash
python main_framework_day7_qwen_lora_eval_diagnosis.py --run_base_comparison --base_eval_samples 128
```

Expected outputs:

```text
data_done/framework_day7_parse_failure_diagnostics.csv
data_done/framework_day7_parse_failure_examples.jsonl
data_done/framework_day7_ranking_case_study.csv
data_done/framework_day7_eval_512_summary.csv
output-repaired/framework/day7_qwen_lora_beauty_listwise_eval512_predictions.jsonl
data_done/framework_day7_prompt_parser_repair_plan.md
data_done/framework_day7_base_vs_lora_comparison.csv
data_done/framework_day7_qwen_lora_eval_diagnosis_report.md
```

Do not retrain, do not run CEP/confidence/evidence framework, and do not commit adapter artifacts or prediction JSONL.
