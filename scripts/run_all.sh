#!/usr/bin/env bash
set -e

echo "========== [1/3] Running clean pipeline =========="
bash scripts/run_clean.sh

echo "========== [2/3] Running noisy pipeline =========="
bash scripts/run_noisy.sh

echo "========== [3/3] Running robustness comparison =========="
bash scripts/run_robustness.sh

echo "========== All done =========="