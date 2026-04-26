# Framework-Day3 Qwen-LoRA Readiness Report

## 1. Why LoRA and Framework Are Separate

LoRA is a way to train a local Qwen3-8B recommendation baseline. The framework contribution is CEP: calibrated relevance posterior, evidence risk, and uncertainty-aware decision mechanisms on top of model/backbone outputs.

## 2. DecodingMatters Reference Points

DecodingMatters shows a practical local LLM recommender pipeline: CSV samples, dataset-to-instruction conversion, HuggingFace Trainer, closed-catalog constrained decoding, and optional logit processing. Its visible training script does not use PEFT/LoRA, so it is an engineering reference rather than the method to copy.

## 3. Our Qwen-LoRA Baseline Task

We generate both listwise closed-candidate ranking and pointwise relevance JSONL. Listwise is the preferred baseline recommender format; pointwise is kept as a bridge to CEP/evidence generator work.

## 4. data_done_lora Outputs

# data_done_lora Stats

| domain | mode | train listwise | train pointwise | valid listwise | valid pointwise | test listwise | test pointwise |
|---|---|---:|---:|---:|---:|---:|---:|
| beauty | full | 622 | 3732 | 622 | 3732 | 622 | 3732 |
| books | sample_1000 | 1000 | 6000 | 1000 | 6000 | 1000 | 6000 |
| electronics | sample_1000 | 1000 | 6000 | 1000 | 6000 | 1000 | 6000 |
| movies | sample_1000 | 1000 | 6000 | 1000 | 6000 | 1000 | 6000 |

## 5. Prompt Templates

Prompts are under `prompts/framework/`. The baseline prompts require JSON output and prohibit candidate IDs outside the candidate pool. The future evidence schema prompt is only a reference and is not used for first-stage baseline training.

## 6. Config Scaffold

`configs/framework/qwen3_8b_lora_baseline_beauty_listwise.yaml` and `configs/framework/qwen3_8b_lora_baseline_beauty_pointwise.yaml` are scaffolds only. The model path is marked TODO and must be verified on the server.

## 7. Next Step

Framework-Day4 should run Qwen tokenizer/inference/parser smoke or implement the LoRA Dataset/Trainer. Do not start large-scale training yet.
