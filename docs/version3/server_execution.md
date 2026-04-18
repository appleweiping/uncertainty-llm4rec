# Version 3 Server-First Execution

This document is the copy-paste oriented server run guide for Version 3.

It assumes:

- the code is already pushed to GitHub
- formal experiments will run on a server
- local Windows runs are only used for smoke tests

## 1. Clone Or Update

First time:

```bash
git clone https://github.com/appleweiping/uncertainty-llm4rec.git
cd uncertainty-llm4rec
```

Existing checkout:

```bash
cd uncertainty-llm4rec
git pull origin main
```

## 2. Create Environment

Recommended Python:

- Python 3.12

Minimal setup:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If local HF inference is part of the run, confirm `torch` and `transformers` are available:

```bash
python - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
```

## 3. Optional API Environment Variables

Set only the providers you actually use:

```bash
export DEEPSEEK_API_KEY="your_key"
export QWEN_API_KEY="your_key"
export GLM_API_KEY="your_key"
export KIMI_API_KEY="your_key"
export DOUBAO_API_KEY="your_key"
```

## 4. Local HF Model Preparation

For local Hugging Face runs, edit the model config if needed:

- `configs/model/qwen_local_7b.yaml`
- `configs/model/llama_local_8b.yaml`

Typical things to verify:

- `local.model_path`
- `local.tokenizer_path`
- `local.device_map`
- `local.dtype`

If the server has a local checkpoint mirror, replace the public model id with the server path.

## 5. Formal Run Templates

### A. Legacy yes/no baseline

Inference:

```bash
python main_infer.py \
  --config configs/exp/beauty_deepseek.yaml \
  --input_path data/processed/amazon_beauty/test.jsonl \
  --output_path outputs/beauty_deepseek/predictions/test_raw.jsonl \
  --split_name test \
  --overwrite
```

Eval:

```bash
python main_eval.py \
  --exp_name beauty_deepseek
```

Calibrate:

```bash
python main_calibrate.py \
  --exp_name beauty_deepseek \
  --valid_path outputs/beauty_deepseek/predictions/valid_raw.jsonl \
  --test_path outputs/beauty_deepseek/predictions/test_raw.jsonl \
  --method isotonic
```

Rerank:

```bash
python main_rerank.py \
  --exp_name beauty_deepseek \
  --task_type pointwise_yesno \
  --input_path outputs/beauty_deepseek/calibrated/test_calibrated.jsonl \
  --lambda_penalty 0.5
```

### B. Local ranking main line

Run the tracked Version 3 ranking template:

```bash
python main_infer.py \
  --config configs/exp/beauty_qwen_local_rank.yaml \
  --overwrite
```

Then evaluate:

```bash
python main_eval.py \
  --exp_name beauty_qwen_local_rank \
  --task_type candidate_ranking \
  --input_path outputs/beauty_qwen_local_rank/predictions/test_ranking_raw.jsonl \
  --k 10
```

Ranking-side uncertainty compare:

```bash
python main_uncertainty_compare.py \
  --exp_name beauty_qwen_local_rank_compare \
  --task_type candidate_ranking \
  --input_path outputs/beauty_qwen_local_rank/predictions/test_ranking_raw.jsonl \
  --output_root outputs \
  --k 10
```

Ranking-aware rerank:

```bash
python main_rerank.py \
  --exp_name beauty_qwen_local_rank_rerank \
  --task_type candidate_ranking \
  --input_path outputs/beauty_qwen_local_rank/predictions/test_ranking_raw.jsonl \
  --output_root outputs \
  --k 10 \
  --lambda_penalty 0.5 \
  --ranking_score_source raw_score \
  --ranking_uncertainty_source inverse_probability
```

### C. Baseline-side confidence validation

Smallest config-driven run:

```bash
python main_baseline_confidence.py \
  --config configs/baseline_confidence/minimal_score_rows.yaml
```

Explicit CLI form:

```bash
python main_baseline_confidence.py \
  --exp_name baseline_confidence_server_smoke \
  --input_path docs/version3/examples/baseline_score_rows_example.jsonl \
  --input_format score_rows \
  --output_root outputs \
  --k 3 \
  --n_bins 5 \
  --high_conf_threshold 0.8 \
  --seed 42
```

## 6. Where Outputs Land

All Version 3 lines still follow the same output layout:

- `outputs/{exp_name}/predictions/`
- `outputs/{exp_name}/calibrated/`
- `outputs/{exp_name}/reranked/`
- `outputs/{exp_name}/figures/`
- `outputs/{exp_name}/tables/`

Examples:

- `outputs/beauty_qwen_local_rank/predictions/test_ranking_raw.jsonl`
- `outputs/beauty_qwen_local_rank_compare/tables/estimator_comparison.csv`
- `outputs/beauty_qwen_local_rank_rerank/tables/rerank_results.csv`
- `outputs/version3_part4_baseline_smoke/tables/baseline_proxy_summary.csv`

## 7. Recommended Order On Server

If you want the safest rollout order:

1. run one legacy pointwise smoke
2. run one local ranking smoke
3. run ranking eval
4. run ranking uncertainty compare
5. run ranking-aware rerank
6. run baseline-side confidence validation

This keeps failures localized and makes it easier to tell whether a problem comes from:

- local backend loading
- ranking prompt / parser
- ranking eval
- ranking uncertainty / rerank
- baseline validation
