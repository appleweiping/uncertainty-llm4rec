# Part 1 Report: Legacy Protection and Local Backend Skeleton

Date: 2026-04-18
Part: Part 1
Theme: legacy yes/no protection + local Hugging Face backend skeleton

## Scope

This stage intentionally touched only the Part 1 boundary:

- protected the existing API-driven pointwise yes/no inference path
- added a local Hugging Face backend skeleton that can be selected through model config

This stage did **not** touch:

- ranking prompts or ranking parsers
- `main_eval.py`
- `main_rerank.py`
- `main_uncertainty_compare.py`
- `src/baselines/*`
- `src/data/*`

For the tightened acceptance standard used later on the same day, the appendix
`2026-04-18_part1_backend_path_proof.md` should be read together with this report.

## Files Changed

Modified:

- `src/llm/base.py`
- `main_infer.py`
- `README.md` (minimal note only)

Added:

- `src/llm/local_backend.py`
- `configs/model/qwen_local_7b.yaml`
- `configs/model/llama_local_8b.yaml`

## What Changed

### 1. Legacy path protection

`main_infer.py` still defaults to the existing `build_backend_from_config()` path for all current API-based model configs. This means:

- existing `deepseek`, `qwen`, `glm`, `kimi`, `doubao` API configs still resolve through the old backend path
- the pointwise yes/no mainline remains the default behavior

### 2. Unified local backend hook

A new `LocalHFBackend` was added in `src/llm/local_backend.py`.

It supports:

- lazy loading of Hugging Face tokenizer/model
- config-driven `model_path`, `tokenizer_path`, `dtype`, `device_map`
- config-driven generation parameters such as `temperature`, `top_p`, and `max_tokens`
- a normalized return structure consistent with the rest of the backend layer

### 3. Unified backend result shape

`src/llm/base.py` now also preserves `backend_type` in the normalized generation result, so downstream logging can distinguish API and local backends without changing the old schema contract for existing consumers.

### 4. Config-selectable local model skeletons

Two local model config templates were added:

- `configs/model/qwen_local_7b.yaml`
- `configs/model/llama_local_8b.yaml`

They are intended as server-side or local-runtime templates, not as hardcoded environment-specific paths.

## Smoke Checks Completed

Completed in this stage:

1. AST/syntax sanity check on:
   - `src/llm/base.py`
   - `src/llm/local_backend.py`
   - `main_infer.py`

2. Legacy backend construction sanity check:
   - `configs/model/deepseek.yaml` still resolves to `DeepSeekBackend`

3. Local backend construction sanity check:
   - `configs/model/qwen_local_7b.yaml` resolves to `LocalHFBackend`
4. Non-sandbox Hugging Face access proof:
   - public tiny model tokenizer download succeeded
   - actual `Qwen/Qwen2.5-7B-Instruct` tokenizer/config access succeeded
5. Runtime-path proof:
   - `main_infer.py` entered `run_pointwise_inference(...)`
   - local config reached `LocalHFBackend.generate(...)`
   - failure occurred inside `LocalHFBackend._load()` during Hugging Face loading

## Recommended Smoke Test Commands

### Legacy yes/no mainline minimal regression

This is the minimal command to verify the old pointwise yes/no inference entry still runs through the existing API backend path:

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek.yaml `
  --input_path data/processed/amazon_beauty/valid.jsonl `
  --output_path outputs/version3_part1_smoke/predictions/valid_raw.jsonl `
  --split_name valid `
  --max_samples 1 `
  --overwrite
```

Preconditions:

- `DEEPSEEK_API_KEY` is set
- network/API access is available

### Local backend minimal smoke test

This is the minimal command to verify that `main_infer.py` can switch to the new local backend path through model config:

```powershell
py -3.12 main_infer.py `
  --config configs/exp/beauty_deepseek.yaml `
  --model_config configs/model/qwen_local_7b.yaml `
  --input_path data/processed/amazon_beauty/valid.jsonl `
  --output_path outputs/version3_part1_local_smoke/predictions/valid_raw.jsonl `
  --split_name valid `
  --max_samples 1 `
  --overwrite
```

Preconditions:

- `transformers` and `torch` are installed
- `configs/model/qwen_local_7b.yaml` points to an available local or accessible Hugging Face checkpoint
- runtime has enough memory/device support for the selected checkpoint

## Current Limitations

- This stage adds only the backend skeleton, not server orchestration or large-scale local experiments.
- No ranking prompt/parser logic was introduced in this stage.
- No baseline-side confidence validation was introduced in this stage.
- The local backend path is wired into `main_infer.py`, and public Hugging Face access is available outside the sandbox.
- The remaining practical limit of this workstation is that the active PyTorch runtime is CPU-only (`torch 2.10.0+cpu`, `cuda_available=False`), so this machine is suitable for path validation and minimal loading checks, but not the intended formal 7B server-first execution target.

## Environment Blockers Explicitly Ruled Out

The following are **not** the current blockers anymore:

- missing `transformers`
- missing `torch`
- invalid public Hugging Face model id for the Qwen local config

The blockers that remain are:

- default sandboxed commands cannot directly reach `huggingface.co`
- this workstation is CPU-only and therefore not the target machine for repeated 7B formal local inference

## Acceptance Status

Under the tightened Part 1 acceptance standard, the status is:

1. old yes/no mainline protection: satisfied
2. local backend is not an empty wire-up: satisfied
3. minimal runnable conditions were tightened as far as this environment allowed:
   - dependencies verified
   - real Hugging Face access verified outside the sandbox
   - actual Qwen repo id verified
4. remaining incomplete part is now a clearly identified environment/runtime blocker:
   - sandbox network restriction during default commands
   - CPU-only workstation for 7B formal execution

So Part 1 is no longer only a structure-level delivery. It is a structure + runtime-path-verified delivery, with the remaining gap isolated to environment limits rather than missing code integration.
