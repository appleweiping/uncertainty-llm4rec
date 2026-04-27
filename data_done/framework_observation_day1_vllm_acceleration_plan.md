# Framework-Observation-Day1 vLLM Acceleration Plan

## Boundary

Transformers remains the reference backend. vLLM is an optional local inference accelerator for the same Qwen3-8B model and the same LoRA adapter. It is not an external API, not a model change, and not a method contribution.

## Proposed Config Fields

```yaml
inference_backend: transformers  # transformers | vllm
vllm_tensor_parallel_size: 1
vllm_gpu_memory_utilization: 0.85
vllm_max_model_len: 2048
vllm_enable_lora: true
vllm_max_loras: 1
vllm_max_lora_rank: 8
```

## Validation Protocol

Use the same Beauty valid 200 samples, same prompt, deterministic generation, and same model/adapter.

Compare:

- parse_success_rate
- schema_valid_rate
- confidence_mean
- confidence_std
- recommend_yes_rate
- runtime_seconds
- rows_per_second

Use `max_new_tokens=96` for the original schema and `max_new_tokens=128` for the refined schema.

## Decision Rule

If vLLM output distribution differs substantially from transformers, do not use vLLM for official observation. If the distributions are consistent and vLLM is faster, vLLM can be used for Beauty full and later four-domain confidence observation.

## Runtime Caution

On a single RTX 4090, do not load transformers and vLLM Qwen processes concurrently. Do not run valid and test with two simultaneous model processes. Let the original transformers full run finish if it is close; otherwise stop gently with Ctrl+C and resume later.
