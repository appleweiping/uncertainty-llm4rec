# Framework-Day5 Server Tiny Train Commands

Run these commands on the server.

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

Make sure Beauty LoRA instruction data exists. If not, regenerate it:

```bash
python main_framework_day45_materialize_lora_data.py --domain beauty --input_root data_done --output_root data_done_lora --mode full --overwrite
```

Optional readiness check:

```bash
python scripts/check_framework_server_readiness.py
```

Run the Day5 tiny Beauty listwise LoRA train:

```bash
python main_framework_day5_tiny_train_qwen_lora.py --config configs/framework/qwen3_8b_lora_baseline_beauty_listwise_tiny.yaml
```

Expected outputs:

```text
data_done/framework_day5_lora_tiny_train_report.md
data_done/framework_day5_lora_tiny_train_metrics.json
artifacts/lora_smoke/qwen3_8b_beauty_listwise_day5_tiny/
```

The `artifacts/` adapter directory is a smoke artifact and should not be committed to GitHub.

If the run OOMs on RTX 4090, first lower `max_seq_len` from `2048` to `1536` in:

```text
configs/framework/qwen3_8b_lora_baseline_beauty_listwise_tiny.yaml
```

Do not change the task logic, prompt, labels, or CEP/framework boundary for Day5.
