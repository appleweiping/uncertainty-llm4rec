# Part 1 Appendix: Local Backend Path Proof

Date: 2026-04-18
Theme: proof that `main_infer.py` actually reaches `LocalHFBackend`

## Goal

This appendix documents the concrete runtime proof that the current Part 1 implementation does not stop at config parsing. When `main_infer.py` is run with:

- `--model_config configs/model/qwen_local_7b.yaml`

the code path really resolves and invokes `LocalHFBackend`, and the failure point in an environment without an available checkpoint happens **inside** the local backend loading/generation path.

This appendix is the runtime-proof companion to:

- `2026-04-18_part1_legacy_and_local_backend.md`

## Exact Call Chain

The current local-backend call chain is:

1. `main_infer.py` parses CLI arguments through `parse_args()`.
2. `main_infer.py` merges CLI/config values through `merge_config()`.
3. `main_infer.py` builds the backend through `build_llm_backend(model_config)`.
4. Inside `build_llm_backend()`:
   - the model yaml is read by `load_yaml(model_cfg_path)`
   - `backend_name` is inspected
   - if `backend_name != "local_hf"`, the old path is preserved and delegated to `build_backend_from_config()`
   - if `backend_name == "local_hf"`, `LocalHFBackend(...)` is instantiated directly
5. Back in `main()`, the resolved backend is printed through `describe_backend(llm_backend)`.
6. `main()` then calls `run_pointwise_inference(...)`.
7. In `src/llm/inference.py`, `run_pointwise_inference(...)` calls:
   - `llm_backend.generate(prompt)`
8. For local configs, this means:
   - `LocalHFBackend.generate(...)`
   - which calls `LocalHFBackend._load()`
   - which tries to load tokenizer/model through Hugging Face `AutoTokenizer.from_pretrained(...)` and `AutoModelForCausalLM.from_pretrained(...)`

This proves the path is:

```text
main_infer.py
  -> build_llm_backend()
  -> LocalHFBackend(...)
  -> run_pointwise_inference()
  -> llm_backend.generate()
  -> LocalHFBackend.generate()
  -> LocalHFBackend._load()
  -> transformers model/tokenizer loading
```

## Runtime Observability Added

`main_infer.py` now prints resolved backend details before the inference loop starts.

The printed structure includes:

- `backend_class`
- `backend_type`
- `provider`
- `model_name`
- `model_path`
- `tokenizer_path`

This makes it easy to distinguish:

- legacy API path
- local Hugging Face path

without reading the code.

## Direct Runtime Proof

The following command was executed:

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

Observed startup output:

```text
[beauty_deepseek] Model config: configs/model/qwen_local_7b.yaml
[beauty_deepseek] Loaded 1 samples.
[beauty_deepseek] Resolved backend: {'backend_class': 'LocalHFBackend', 'backend_type': 'local_hf', 'provider': 'local_hf', 'model_name': 'qwen2.5-7b-instruct-local', 'model_path': 'Qwen/Qwen2.5-7B-Instruct', 'tokenizer_path': 'Qwen/Qwen2.5-7B-Instruct'}
```

After this, the program entered:

- `run_pointwise_inference(...)`
- `llm_backend.generate(prompt)`
- `LocalHFBackend.generate(...)`
- `LocalHFBackend._load()`

and then failed during:

- `AutoTokenizer.from_pretrained(...)`

This is the expected proof point: the code did not stop at config parsing or backend construction. It reached the actual local backend loading logic.

## Additional Environment Tightening

After the first failure, the environment was checked more carefully instead of stopping at a theoretical explanation.

### Dependency status

The active Python 3.12 runtime already has the required local-backend packages:

- `transformers 5.5.3`
- `torch 2.10.0+cpu`

So dependency absence is **not** the current blocker.

### Hugging Face connectivity status

Under the default sandboxed command environment, direct access to `huggingface.co` was refused with:

- `httpx.ConnectError`
- `[WinError 10061]`

After retrying outside the sandbox, public Hugging Face access succeeded:

1. Tiny public smoke access succeeded:
   - `AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')`
2. The actual Qwen config target also succeeded for minimal remote resolution:
   - `AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)`
   - `AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)`

This means:

- the local backend code path is real
- public Hugging Face download/connectivity is available when sandbox network restrictions are removed
- the earlier connection failure was an environment restriction of the default sandboxed execution path, not a fake backend path

### What was actually completed versus what was only inferred

Completed directly:

- backend resolution to `LocalHFBackend`
- execution into `run_pointwise_inference(...)`
- execution into `LocalHFBackend.generate(...)`
- execution into `LocalHFBackend._load(...)`
- public Hugging Face access for a tiny model
- public Hugging Face access for the actual Qwen repo used by the local config

Not completed on this workstation:

- full 7B weight loading to a formal local inference runtime
- repeated 7B generation as a production-like server-first path

The reason is not missing code integration. The remaining reason is runtime suitability of the current machine.

## Failure Position and Error Type

### In the current environment

The current environment has the required local-backend Python packages available, so dependency installation is not the blocker.

Actual failure location:

- `src/llm/local_backend.py`
- inside `LocalHFBackend._load()`
- at `AutoTokenizer.from_pretrained(...)`

Actual top-level failure type:

- `OSError`

Observed failure meaning:

- in the sandboxed run, the environment could not connect to `huggingface.co`
- therefore the configured repo id could not be resolved in that execution mode

This is exactly the kind of error we want to see for proof purposes, because it confirms that the runtime already entered the real local generation path.

### Remaining practical constraint for full 7B generation

After confirming that:

- dependencies exist
- public Hugging Face access works outside the sandbox
- the actual Qwen repo id is reachable

the remaining practical limitation of this machine is that the installed PyTorch build is:

- `torch 2.10.0+cpu`

and CUDA is unavailable:

- `torch.cuda.is_available() == False`

So this workstation is able to prove:

- backend construction
- prompt-to-generate call path
- tokenizer/config access to the real Qwen repo

but it is not a suitable formal runtime for repeated local 7B server-first experiments. That role should be assigned to the actual server/GPU environment in later parts.

### If `transformers` / `torch` were missing

Then the failure would happen earlier, still inside:

- `LocalHFBackend._load()`

and the error type would be:

- `ImportError`

with the explicit message raised by this backend:

```text
LocalHFBackend requires `transformers` and `torch`. Install them before using backend_name=local_hf.
```

This also counts as valid proof that the program reached the local backend implementation.

### If `model_path` were empty

Then the failure would happen in:

- `build_llm_backend()` in `main_infer.py`

with:

- `ValueError`

because local configs are required to define `model_path`.

This is the only case that fails before entering `LocalHFBackend.generate()`, and it is a config-definition error rather than a fake runtime path.

## Acceptance Conclusion

For Part 1 acceptance, the relevant proof is now available:

1. `main_infer.py` visibly resolves `LocalHFBackend` at startup
2. the inference loop really calls `llm_backend.generate(...)`
3. the runtime really enters `LocalHFBackend.generate()` and `LocalHFBackend._load()`
4. in the absence of an available checkpoint, the failure occurs in Hugging Face model/tokenizer loading, not in config parsing

This means the local backend is not an empty shell and not a config-only stub. It is already wired into the actual inference path.
