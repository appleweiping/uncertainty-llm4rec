# Week7 Local Backend And Server Workflow

Week7 changes the execution role of the project. Official APIs remain useful for small cross-model observations, case studies, and consistency checks, but they should not be treated as the main experiment throughput path. The main experiment path is now the server-side Hugging Face backend, with the base model stored on the execution machine and the code synchronized through Git.

The selected Week7 main local model is Llama 3.1 8B Instruct. The default config is `configs/model/llama31_8b_instruct_local.yaml`, which points to `/home/ajifang/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct`. If the server image mounts the model workspace under another path, adjust only `model_name_or_path` and `tokenizer_name_or_path`; do not hard-code SSH credentials or passwords into configs, docs, scripts, or logs.

The intended workflow is:

1. Keep code editing and Git commits on the local workstation.
2. Push the branch to GitHub.
3. On the server, pull the branch into the project checkout.
4. Store the base model in the server model directory or Hugging Face cache.
5. Run `main_backend_check.py` first to verify local model loading and the output schema.
6. Run task-specific smoke configs before medium-scale experiments.
7. Keep future LoRA adapters as separate server-side adapter directories rather than duplicating the base model.

The backend abstraction is centered on `src/llm/base.py`. API backends and local HF backends expose the same `generate()` and `batch_generate()` shape so that pointwise, pairwise, and candidate ranking inference can keep using the existing parser and evaluation stack.

The local HF implementation lives in `src/llm/local_hf_backend.py`. It supports local model paths, tokenizer paths, `device_map`, dtype, batch size, local-files-only loading, optional 4-bit or 8-bit flags, optional adapter path, and chat-template formatting. This is intentionally a base-only first path: LoRA and vLLM should be added after the server-side base inference path is stable.

Minimal server checks:

```bash
python main_backend_check.py \
  --model_config configs/model/llama31_8b_instruct_local.yaml \
  --status_path outputs/summary/week7_day1_backend_check.csv
```

or, from the repository root on the server:

```bash
bash scripts/week7_day1_server_backend_check.sh "$PWD"
```

Smoke task configs:

```bash
python main_infer.py --config configs/exp/beauty_llama31_local_pointwise_smoke.yaml
python main_rank.py --config configs/exp/beauty_llama31_local_rank_smoke.yaml
python main_pairwise.py --config configs/exp/beauty_llama31_local_pairwise_smoke.yaml
```

To run the backend check and the three smoke configs together:

```bash
RUN_SMOKE=1 bash scripts/week7_day1_server_backend_check.sh "$PWD"
```

The local workstation can run a dry-run status file without loading the model:

```bash
python main_backend_check.py \
  --model_config configs/model/llama31_8b_instruct_local.yaml \
  --status_path outputs/summary/week7_day1_backend_check.csv \
  --dry_run
```

This dry run only confirms config wiring. The actual Week7 Day1 acceptance check requires the non-dry-run command on the Ubuntu 22 + RTX 4090 server.
