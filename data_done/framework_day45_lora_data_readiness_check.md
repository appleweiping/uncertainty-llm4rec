# Framework-Day4.5 LoRA Data Readiness Check

## Status

- Status: `blocked`
- Blocking reasons: `missing_data_done_beauty_train, missing_data_done_lora_jsonl, config_points_to_missing_lora_files`

## Source data_done

- `data_done/beauty/train.jsonl` exists: `False`
- rows: `None`

## data_done_lora Files

- `data_done_lora/beauty/train_listwise.jsonl`: exists=`False`, rows=`None`
- `data_done_lora/beauty/valid_listwise.jsonl`: exists=`False`, rows=`None`
- `data_done_lora/beauty/test_listwise.jsonl`: exists=`False`, rows=`None`
- `data_done_lora/beauty/train_pointwise.jsonl`: exists=`False`, rows=`None`
- `data_done_lora/beauty/valid_pointwise.jsonl`: exists=`False`, rows=`None`
- `data_done_lora/beauty/test_pointwise.jsonl`: exists=`False`, rows=`None`

## Config

- config: `configs/framework/qwen3_8b_lora_baseline_beauty_listwise.yaml`
- train_file exists: `False` (`data_done_lora/beauty/train_listwise.jsonl`)
- valid_file exists: `False` (`data_done_lora/beauty/valid_listwise.jsonl`)
- model path TODO: `False`
- model path exists: `True`

## Environment

- torch importable: `True`
- transformers importable: `True`
- peft importable: `True`
- accelerate importable: `True`
- bitsandbytes importable: `False`
- cuda available: `True`
