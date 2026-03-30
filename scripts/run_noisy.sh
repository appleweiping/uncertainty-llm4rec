#!/usr/bin/env bash
set -e

python scripts/generate_noisy_data.py \
  --input_path data/processed/test.jsonl \
  --output_path data/processed/test_noisy.jsonl \
  --metadata_path data/processed/test_noisy_metadata.json \
  --history_drop_prob 0.2 \
  --text_noise_prob 0.5 \
  --label_flip_prob 0.0 \
  --seed 42

python main_infer.py --exp_name noisy --overwrite
python main_eval.py --exp_name noisy
python main_calibrate.py --exp_name noisy
python main_rerank.py --exp_name noisy --lambda_penalty 0.5