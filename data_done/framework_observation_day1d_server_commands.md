# Framework-Observation-Day1d Server Commands

## Pull Latest

```bash
cd /home/ajifang/projects/uncertainty-llm4rec-week4
git pull origin codex/week4-confidence-repair
```

## Activate Environment

Day1d uses transformers for logit extraction, not vLLM logprobs.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm

python - <<'PY'
import torch, transformers, peft, sklearn
print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("sklearn", sklearn.__version__)
PY
```

## Run Day1d Beauty 200/200

Run valid and test sequentially, not concurrently:

```bash
python main_framework_observation_day1d_logit_confidence.py --config configs/framework_observation/beauty_qwen_lora_logit_confidence.yaml --split valid --model_variant lora --max_samples 200 --resume

python main_framework_observation_day1d_logit_confidence.py --config configs/framework_observation/beauty_qwen_lora_logit_confidence.yaml --split test --model_variant lora --max_samples 200 --resume

python main_framework_observation_day1d_logit_confidence.py --config configs/framework_observation/beauty_qwen_lora_logit_confidence.yaml --analyze_only
```

## Server Packaging

```bash
python scripts/framework_artifact_manifest.py

tar -czf framework_observation_day1d_logit_confidence_results.tar.gz \
  data_done/framework_observation_day1d_logit_confidence_diagnostics.csv \
  data_done/framework_observation_day1d_logit_confidence_calibration.csv \
  data_done/framework_observation_day1d_logit_confidence_report.md \
  data_done/framework_observation_day1_prompt_comparison.csv \
  data_done/framework_artifact_manifest_server.json \
  data_done/framework_artifact_manifest_server.md \
  output-repaired/framework_observation/beauty_qwen_lora_logit_confidence/predictions/valid_raw.jsonl \
  output-repaired/framework_observation/beauty_qwen_lora_logit_confidence/predictions/test_raw.jsonl
```

## Local Pull

```powershell
cd D:\Research\Uncertainty-LLM4Rec

scp ajifang@125.71.97.70:/home/ajifang/projects/uncertainty-llm4rec-week4/framework_observation_day1d_logit_confidence_results.tar.gz .
tar -xzf framework_observation_day1d_logit_confidence_results.tar.gz
```
