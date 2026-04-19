# Week7 Day2 Batch Workflow

Week7 Day2 turns the local-HF backend handoff into a structured experiment queue. The goal is not to run a large matrix from the local workstation, but to make the server-side run path explicit enough that pointwise, candidate ranking, and pairwise experiments can be launched, tracked, retried, and summarized through one registry.

The main entry point is:

```bash
python main_batch_run.py --batch_config configs/batch/week7_local_scale.yaml
```

This default call is intentionally safe. It does not launch model inference; it validates the batch specification, checks whether inputs exist, builds the exact commands, and writes `outputs/summary/week7_day2_batch_status.csv`. Real GPU execution requires an explicit `--run` flag:

```bash
PYTHON_BIN=/path/to/env/bin/python python main_batch_run.py \
  --batch_config configs/batch/week7_local_scale.yaml \
  --run
```

The batch config stores a small but complete local-HF smoke queue: Beauty pointwise, Beauty candidate ranking, and Beauty pairwise preference. Each experiment records `exp_name`, `domain`, `task`, `model`, `method_family`, `method_variant`, `config_path`, `output_dir`, status, latency, logs, return code, and error message. This is deliberately stricter than a shell loop because later Week8 experiments will need task/model/domain/method identity to remain recoverable after partial failures.

Failure recovery starts from the registry instead of memory. After a failed run, inspect `outputs/summary/week7_day2_batch_status.csv` and the per-experiment logs under `outputs/logs/batch/week7_local_scale/`. To rerun only failed rows, use:

```bash
PYTHON_BIN=/path/to/env/bin/python python main_batch_run.py \
  --batch_config configs/batch/week7_local_scale.yaml \
  --only_failed \
  --run
```

This day is still a framework day. It does not open new ranking families, does not move `local_margin_swap_rerank` or the fully fused variant back into the default line, does not start LoRA, and does not run the Week8 matrix. The current default method family remains the structured-risk line from Week6; the batch layer only makes future execution less fragile.

