# Framework-Day6 Beauty Listwise Small Train Report

## Scope

This records the successful server-side Beauty listwise Qwen-LoRA small train. It is still a baseline infrastructure run: no confidence module, no evidence module, no CEP fusion, no API calls, and no four-domain training.

## Train Result

- status: `success`
- max steps: `100`
- train samples: `512`
- eval samples: `128`
- loss NaN count: `0`
- peak GPU memory: `20.2553 GB`
- runtime: `325.72 seconds`
- adapter output directory: `artifacts/lora/qwen3_8b_beauty_listwise_day6_small`

## Artifact Policy

The adapter path is an ignored server artifact and must not be committed to GitHub.

## Interpretation

Day6 proves the Qwen3-8B + LoRA baseline can complete train, save adapter, load adapter, infer, parse, and evaluate. It does not yet prove strong baseline performance.
