# Framework-Day4.5 Server Materialization Instructions

Run these commands on the server after pulling the updated branch.

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

If the config still contains `TODO_MODEL_PATH`, update both fields in:

```text
configs/framework/qwen3_8b_lora_baseline_beauty_listwise.yaml
configs/framework/qwen3_8b_lora_baseline_beauty_pointwise.yaml
```

to:

```text
/home/ajifang/models/Qwen/Qwen3-8B
```

Materialize Beauty LoRA instruction JSONL from `data_done/`:

```bash
python main_framework_day45_materialize_lora_data.py --domain beauty --input_root data_done --output_root data_done_lora --mode full --overwrite
```

Check data/config/server readiness:

```bash
python scripts/check_framework_server_readiness.py
```

Then rerun the Day4 dry-run:

```bash
python main_framework_day4_train_qwen_lora_baseline.py --config configs/framework/qwen3_8b_lora_baseline_beauty_listwise.yaml --dry_run
```

Expected materialized Beauty files:

```text
data_done_lora/beauty/train_listwise.jsonl
data_done_lora/beauty/valid_listwise.jsonl
data_done_lora/beauty/test_listwise.jsonl
data_done_lora/beauty/train_pointwise.jsonl
data_done_lora/beauty/valid_pointwise.jsonl
data_done_lora/beauty/test_pointwise.jsonl
```

These JSONL files are generated data artifacts and are intentionally not committed to GitHub.
