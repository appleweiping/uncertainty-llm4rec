#!/usr/bin/env bash
set -e

echo "======================================"
echo " Running CLEAN experiment (MovieLens 1M) "
echo "======================================"

python main_infer.py \
  --exp_name clean \
  --input_path data/processed/movielens_1m/test.jsonl \
  --overwrite

python main_eval.py \
  --exp_name clean

python main_calibrate.py \
  --exp_name clean

python main_rerank.py \
  --exp_name clean \
  --lambda_penalty 0.5

echo "======================================"
echo " CLEAN experiment finished "
echo "======================================"