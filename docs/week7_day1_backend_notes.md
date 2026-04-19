# Week7 Day1 Backend Notes

The Week7 Day1 backend line is now defined around a two-layer execution model. API backends remain available for observation and external comparison, while the main experiment throughput should move to the server-side Hugging Face backend.

Current selected server setting:

- OS: Ubuntu 22
- GPU: RTX 4090 class, 48G memory
- Main local model: Llama 3.1 8B Instruct
- Default config: `configs/model/llama31_8b_instruct_local.yaml`
- Default model path: `/home/ajifang/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct`

No server password, remote desktop password, SSH password, token, or temporary URL should be committed to this repository.

The local HF backend is implemented in `src/llm/local_hf_backend.py` and registered through `src/llm/__init__.py` under `local_hf`, `hf`, and `transformers`. It uses the same inference-level shape as API backends, so `main_infer.py`, `main_rank.py`, and `main_pairwise.py` can continue to load the backend from `model_config`.

The first server-side check should run:

```bash
python main_backend_check.py \
  --model_config configs/model/llama31_8b_instruct_local.yaml \
  --status_path outputs/summary/week7_day1_backend_check.csv
```

The equivalent server-side helper is:

```bash
bash scripts/week7_day1_server_backend_check.sh "$PWD"
```

Set `RUN_SMOKE=1` before the helper only after the model path and Python dependencies are confirmed.

The three smoke configs are:

- `configs/exp/beauty_llama31_local_pointwise_smoke.yaml`
- `configs/exp/beauty_llama31_local_rank_smoke.yaml`
- `configs/exp/beauty_llama31_local_pairwise_smoke.yaml`

Current boundary: this day establishes the backend abstraction and server workflow. It does not run medium-scale experiments, does not train LoRA, does not start vLLM serving, and does not change the Part5 structured-risk method family.
