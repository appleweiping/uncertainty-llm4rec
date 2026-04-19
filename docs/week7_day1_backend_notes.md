# Week7 Day1 Backend Notes

The Week7 Day1 backend line is now defined around a two-layer execution model. API backends remain available for observation and external comparison, while the main experiment throughput should move to the server-side Hugging Face backend.

Current selected server setting:

- OS: Ubuntu 22
- GPU: NVIDIA GeForce RTX 4090, 49140 MiB memory
- NVIDIA driver: 570.211.01
- CUDA runtime reported by `nvidia-smi`: 12.8
- Default `python3`: 3.13.5
- Default user workspace: `/home/ajifang`
- Main local model: Qwen3-8B
- Default config: `configs/model/qwen3_8b_local.yaml`
- Verified model path on the current server: `/home/ajifang/models/Qwen/Qwen3-8B`
- Download source: ModelScope, with runtime loading from the server-local path

No server password, remote desktop password, SSH password, token, or temporary URL should be committed to this repository.

The model path above is a server environment correction, not a method change. The earlier placeholder path was replaced because the current server has already verified local Qwen3 loading from `/home/ajifang/models/Qwen/Qwen3-8B`.

The server already has a recent CUDA/driver stack, so the main remaining runtime risk is the Python environment. Because the default `python3` is 3.13.5, the first real run should happen inside a pinned conda environment after PyTorch, Transformers, Accelerate, and optional quantization dependencies are confirmed. The helper script supports `PYTHON_BIN=...` so the experiment runner can point to the intended environment without editing repository files.

The local HF backend is implemented in `src/llm/local_hf_backend.py` and registered through `src/llm/__init__.py` under `local_hf`, `hf`, and `transformers`. It uses the same inference-level shape as API backends, so `main_infer.py`, `main_rank.py`, and `main_pairwise.py` can continue to load the backend from `model_config`.

Qwen3 can emit `<think>...</think>` reasoning blocks by default. The current server-local route therefore treats thinking control as an execution-layer compatibility issue, not a method change: `configs/model/qwen3_8b_local.yaml` sets `enable_thinking: false`, the local HF backend forwards that option to chat templates that support it, the prompt templates explicitly request final JSON only, and `src/llm/parser.py` strips any remaining thinking block before task-specific parsing. This shared cleanup guard covers pointwise, candidate ranking, and pairwise preference outputs.

The first server-side check should run:

```bash
python main_backend_check.py \
  --model_config configs/model/qwen3_8b_local.yaml \
  --status_path outputs/summary/week7_day1_backend_check.csv
```

The equivalent server-side helper is:

```bash
bash scripts/week7_day1_server_backend_check.sh "$PWD"
```

If the conda environment exposes a different interpreter, run:

```bash
PYTHON_BIN=/path/to/env/bin/python bash scripts/week7_day1_server_backend_check.sh "$PWD"
```

Set `RUN_SMOKE=1` before the helper only after the model path and Python dependencies are confirmed.

The three smoke configs are:

- `configs/exp/beauty_qwen3_local_pointwise_smoke.yaml`
- `configs/exp/beauty_qwen3_local_rank_smoke.yaml`
- `configs/exp/beauty_qwen3_local_pairwise_smoke.yaml`

Current boundary: this day establishes the backend abstraction and server workflow. It does not run medium-scale experiments, does not train LoRA, does not start vLLM serving, and does not change the Part5 structured-risk method family.
