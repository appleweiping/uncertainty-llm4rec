# Framework-Day4.5 LoRA Data Materialization Report

## Why Server Day4 Dry-Run Failed

The server had `data_done_lora/beauty` metadata files but not the large instruction JSONL files. Those JSONL files are intentionally git-ignored and were not committed, so the Day4 dataset loader could not find `train_listwise.jsonl`.

## Missing Files Repaired by Materialization

- `train_listwise.jsonl`
- `valid_listwise.jsonl`
- `test_listwise.jsonl`
- `train_pointwise.jsonl`
- `valid_pointwise.jsonl`
- `test_pointwise.jsonl`

## Script

`main_framework_day45_materialize_lora_data.py` regenerates the JSONL files from `data_done/{domain}` with seed=42. It does not use calibrated probability, confidence, evidence, or CEP fields as labels.

## Beauty Expected Rows

- train listwise: `622`
- train pointwise: `3732`
- valid listwise: `622`
- valid pointwise: `3732`
- test listwise: `622`
- test pointwise: `3732`

## Not Committed to GitHub

The generated `data_done_lora/**/*.jsonl` files are data artifacts and remain git-ignored. Commit scripts, configs, manifests, and reports only.

## Server Regeneration

Run the command in `data_done/framework_day45_server_materialize_lora_data_instructions.md` after pulling the branch.

## Day5 Readiness

After materialization, Day5 can continue with Qwen tokenizer/model forward smoke and then a tiny LoRA train if the server model path and GPU remain available.
