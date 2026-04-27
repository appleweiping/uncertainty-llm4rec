# Framework-Observation-Day1c Server Commands

## Pull Latest

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

## Activate vLLM Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm

python - <<'PY'
import torch, transformers, vllm
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY
```

## Ensure GPU Is Free

```bash
nvidia-smi
ps -u ajifang -f | grep -E "python|vllm|EngineCore" | grep -v grep
```

Stop only stale `VLLM::EngineCore` or old Qwen Python processes. Do not stop unrelated long-running CPU jobs unless explicitly intended.

## Run Day1c 200/200 Smoke

Run valid and test sequentially, not concurrently:

```bash
python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence_decision_forced.yaml --split valid --model_variant lora --inference_backend vllm --max_samples 200 --resume

python main_framework_observation_day1_local_confidence_infer.py --config configs/framework_observation/beauty_qwen_lora_confidence_decision_forced.yaml --split test --model_variant lora --inference_backend vllm --max_samples 200 --resume

python main_framework_observation_day1_confidence_analysis.py --pred_dir output-repaired/framework_observation/beauty_qwen_lora_confidence_decision_forced/predictions --variant decision_forced
```

## Sync Lightweight Results

```bash
python scripts/framework_artifact_manifest.py

tar -czf framework_observation_day1c_decision_forced_results.tar.gz \
  data_done/framework_observation_day1c_decision_forced_confidence_diagnostics.csv \
  data_done/framework_observation_day1c_decision_forced_calibration.csv \
  data_done/framework_observation_day1c_decision_forced_report.md \
  data_done/framework_observation_day1_prompt_comparison.csv \
  data_done/framework_artifact_manifest_server.json \
  data_done/framework_artifact_manifest_server.md \
  output-repaired/framework_observation/beauty_qwen_lora_confidence_decision_forced/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_confidence_decision_forced/predictions/test_raw.jsonl
```
