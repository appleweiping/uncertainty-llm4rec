#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$PWD}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model/llama31_8b_instruct_local.yaml}"
STATUS_PATH="${STATUS_PATH:-outputs/summary/week7_day1_backend_check.csv}"
RUN_SMOKE="${RUN_SMOKE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_DIR"

"$PYTHON_BIN" main_backend_check.py \
  --model_config "$MODEL_CONFIG" \
  --status_path "$STATUS_PATH"

if [[ "$RUN_SMOKE" == "1" ]]; then
  "$PYTHON_BIN" main_infer.py --config configs/exp/beauty_llama31_local_pointwise_smoke.yaml
  "$PYTHON_BIN" main_rank.py --config configs/exp/beauty_llama31_local_rank_smoke.yaml
  "$PYTHON_BIN" main_pairwise.py --config configs/exp/beauty_llama31_local_pairwise_smoke.yaml
fi

echo "Week7 Day1 server backend check finished."
