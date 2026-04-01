#!/usr/bin/env bash
set -e

echo "======================================"
echo " Generating NOISY data (MovieLens 1M) "
echo "======================================"

python scripts/generate_noisy_data.py \
  --input_path data/processed/movielens_1m/test.jsonl \
  --output_path data/processed/movielens_1m/test_noisy.jsonl \
  --metadata_path data/processed/movielens_1m/test_noisy_metadata.json \
  --history_drop_prob 0.2 \
  --text_noise_prob 0.5 \
  --label_flip_prob 0.0 \
  --seed 42

echo "======================================"
echo " Running NOISY experiment "
echo "======================================"

python main_infer.py \
  --exp_name noisy \
  --input_path data/processed/movielens_1m/test_noisy.jsonl \
  --overwrite

python main_eval.py \
  --exp_name noisy

python main_calibrate.py \
  --exp_name noisy

python main_rerank.py \
  --exp_name noisy \
  --lambda_penalty 0.5

echo "======================================"
echo " NOISY experiment finished "
echo "======================================"