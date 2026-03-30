#!/usr/bin/env bash
set -e

python main_infer.py --exp_name clean --overwrite
python main_eval.py --exp_name clean
python main_calibrate.py --exp_name clean
python main_rerank.py --exp_name clean --lambda_penalty 0.5