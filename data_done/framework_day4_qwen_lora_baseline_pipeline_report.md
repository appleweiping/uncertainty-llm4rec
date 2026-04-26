# Framework-Day4 Qwen-LoRA Baseline Pipeline Report

## 1. Day3 Recap

Framework-Day3 prepared `data_done_lora`, baseline prompts, and Qwen3-8B LoRA config scaffolds for Beauty listwise and pointwise tasks.

## 2. LoRA Baseline vs Framework Boundary

LoRA is only the local Qwen3-8B recommendation baseline. CEP/calibrated posterior/evidence risk are not trained into this baseline and remain later decision-stage framework modules.

## 3. Future Confidence + Evidence Direction

The future framework will combine calibrated confidence uncertainty from the week1-week4 confidence line and evidence risk / calibrated relevance posterior from the Day9+ evidence line. Day4 does not implement that fusion.

## 4. Dataset Loader

`src/framework/lora_dataset.py` reads listwise and pointwise JSONL, formats samples, masks labels, preserves metadata, and excludes calibrated probability / CEP fields from targets.

## 5. Prompt Formatter

`src/framework/prompt_formatters.py` provides stable Qwen baseline formatting for closed-candidate ranking and pointwise relevance.

## 6. Training Entrypoint

`main_framework_day4_train_qwen_lora_baseline.py` reads config, checks dependencies/model path/GPU, builds datasets, supports dry-run, and can run a forward smoke only when model path and GPU are available.

## 7. Dry-Run Result

- Status: `blocked`
- Blocked reasons: `dependency_missing:peft, model_path_missing`
- Label mask check: `skipped`
- Dataset stats: `data_done/framework_day4_lora_dataset_stats.csv`
- Sample prompts: `data_done/framework_day4_lora_sample_prompts.json`

## 8. Current Ready / Blocked State

The local dataset/prompt/config pipeline is ready. Full model smoke/training is blocked locally if `model_path_missing`, `gpu_missing`, or dependency issues are listed above. This is environment readiness, not method failure.

## 9. Day5 Recommendation

If the server model path and GPU are available, run a Beauty listwise tiny LoRA train. If not, first fix the server environment. Do not enter confidence/evidence framework fusion before baseline training smoke is stable.
