# Local vLLM Environment Note

## Stable Environment

Use the dedicated conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm
```

Expected core versions:

```text
python 3.12.x
torch 2.8.0+cu128
CUDA runtime 12.8
vllm 0.10.2
transformers 4.55.2
tokenizers 0.21.4
huggingface-hub 0.36.0
```

Quick check:

```bash
python - <<'PY'
import torch, transformers, vllm
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY
```

## Do Not Auto-Upgrade

Do not run unconstrained upgrades inside `qwen_vllm`, especially:

```bash
pip install -U torch transformers peft vllm
pip install peft transformers accelerate
```

Those commands can pull CUDA 13 wheels or incompatible Transformers 5.x packages and break vLLM on the current NVIDIA driver.

If a dependency is missing, install only that package and avoid changing torch:

```bash
python -m pip install --no-cache-dir --no-deps PACKAGE_NAME
```

## Known Working Constraints

- The server driver reports CUDA 12.8 support, so `torch 2.8.0+cu128` is the safe target.
- vLLM initialization can reserve a large KV cache. Do not run two Qwen/vLLM processes at the same time on the single 4090.
- If vLLM reports low free GPU memory, check and stop stale `VLLM::EngineCore` processes before retrying.

```bash
nvidia-smi
ps -u ajifang -f | grep -E "python|vllm|EngineCore" | grep -v grep
```

## Day1c Command Pattern

```bash
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence_decision_forced.yaml --split valid --model_variant lora --inference_backend vllm --max_samples 200 --resume
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence_decision_forced.yaml --split test --model_variant lora --inference_backend vllm --max_samples 200 --resume
python main_framework_observation_day1_confidence_analysis.py --pred_dir output-repaired/framework_observation/beauty_qwen_lora_confidence_decision_forced/predictions --variant decision_forced
```
