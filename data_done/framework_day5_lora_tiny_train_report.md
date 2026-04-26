# Framework-Day5 Beauty Listwise Qwen-LoRA Tiny Train Report

## Scope

This records the successful server-side tiny LoRA training smoke for the Beauty listwise Qwen3-8B baseline. It does not implement confidence, evidence, CEP fusion, API calls, or formal long training.

## Server Environment

- Project directory: `/home/ajifang/projects/uncertainty-llm4rec-week4`
- Model path: `/home/ajifang/models/Qwen/Qwen3-8B`
- GPU: RTX 4090
- CUDA available: `true`
- Dependencies available: `torch`, `transformers`, `peft`, `accelerate`
- `bitsandbytes`: unavailable, not required by the Day5 script

## Tiny Train Parameters

- task: `candidate_ranking_listwise`
- train samples: `64`
- eval samples: `64`
- max steps: `10`
- batch size: `1`
- gradient accumulation steps: `4`
- max sequence length: `2048`
- LoRA rank / alpha / dropout: `8 / 16 / 0.05`

## Result

- status: `success`
- loss first: `0.4933`
- loss last: `0.0090`
- loss NaN count: `0`
- peak GPU memory: `20.2543 GB`
- runtime: `32.62 seconds`
- OOM status: `false`
- ready for Day6 real train: `true`

## Adapter Artifact

- adapter output directory: `artifacts/lora_smoke/qwen3_8b_beauty_listwise_day5_tiny`
- commit policy: adapter/checkpoint artifacts are ignored and must not be committed to GitHub

## Interpretation

Day5 confirms that the Qwen3-8B LoRA baseline infrastructure can perform real optimizer steps on the server. This is a training-pipeline smoke, not a recommendation performance claim and not a CEP/framework result.
