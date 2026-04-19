# Week7 Local Backend And Server Workflow

Week7 changes the execution role of the project. Official APIs remain useful for small cross-model observations, case studies, and consistency checks, but they should not be treated as the main experiment throughput path. The main experiment path is now the server-side Hugging Face backend, with the base model stored on the execution machine and the code synchronized through Git.

The selected Week7 main local model is now Qwen3-8B. The default config is `configs/model/qwen3_8b_local.yaml`, which points to `/home/ajifang/autodl-tmp/models/Qwen3-8B`. The model should be downloaded or mirrored through ModelScope, then loaded from that server-local path with `local_files_only: true`; runtime inference should not depend on online Hugging Face access. The currently verified server baseline is Ubuntu 22 with an RTX 4090-class 48G GPU, NVIDIA driver 570.211.01, CUDA 12.8 as reported by `nvidia-smi`, and `/home/ajifang` as the user workspace. If the server image mounts the model workspace under another path, adjust only `model_name_or_path` and `tokenizer_name_or_path`; do not hard-code SSH credentials or passwords into configs, docs, scripts, or logs.

The intended workflow is:

1. Keep code editing and Git commits on the local workstation.
2. Push the branch to GitHub.
3. On the server, pull the branch into the project checkout.
4. Store the base model in the server model directory after downloading through ModelScope.
5. Run `main_backend_check.py` first to verify local model loading and the output schema.
6. Run task-specific smoke configs before medium-scale experiments.
7. Keep future LoRA adapters as separate server-side adapter directories rather than duplicating the base model.

The default ModelScope preparation command is:

```bash
modelscope download --model Qwen/Qwen3-8B --local_dir /home/ajifang/autodl-tmp/models/Qwen3-8B
```

The backend abstraction is centered on `src/llm/base.py`. API backends and local HF backends expose the same `generate()` and `batch_generate()` shape so that pointwise, pairwise, and candidate ranking inference can keep using the existing parser and evaluation stack.

The local HF implementation lives in `src/llm/local_hf_backend.py`. It supports local model paths, tokenizer paths, `device_map`, dtype, batch size, local-files-only loading, optional 4-bit or 8-bit flags, optional adapter path, and chat-template formatting. This is intentionally a base-only first path: LoRA and vLLM should be added after the server-side base inference path is stable.

Minimal server checks:

```bash
python main_backend_check.py \
  --model_config configs/model/qwen3_8b_local.yaml \
  --status_path outputs/summary/week7_day1_backend_check.csv
```

or, from the repository root on the server:

```bash
bash scripts/week7_day1_server_backend_check.sh "$PWD"
```

Smoke task configs:

```bash
python main_infer.py --config configs/exp/beauty_qwen3_local_pointwise_smoke.yaml
python main_rank.py --config configs/exp/beauty_qwen3_local_rank_smoke.yaml
python main_pairwise.py --config configs/exp/beauty_qwen3_local_pairwise_smoke.yaml
```

To run the backend check and the three smoke configs together:

```bash
RUN_SMOKE=1 bash scripts/week7_day1_server_backend_check.sh "$PWD"
```

If the default `python3` is not the intended experiment environment, select the interpreter explicitly:

```bash
PYTHON_BIN=/path/to/env/bin/python RUN_SMOKE=1 bash scripts/week7_day1_server_backend_check.sh "$PWD"
```

The local workstation can run a dry-run status file without loading the model:

```bash
python main_backend_check.py \
  --model_config configs/model/qwen3_8b_local.yaml \
  --status_path outputs/summary/week7_day1_backend_check.csv \
  --dry_run
```

This dry run only confirms config wiring. The actual Week7 Day1 acceptance check requires the non-dry-run command on the Ubuntu 22 + RTX 4090 server.
