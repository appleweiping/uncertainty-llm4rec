#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$PWD}"
BATCH_CONFIG="${BATCH_CONFIG:-configs/batch/week7_local_scale.yaml}"
STATUS_PATH="${STATUS_PATH:-outputs/summary/week7_day2_batch_status.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PULL_FIRST="${PULL_FIRST:-1}"
RUN_BATCH="${RUN_BATCH:-1}"
ONLY_FAILED="${ONLY_FAILED:-0}"

cd "$REPO_DIR"

if [[ "$PULL_FIRST" == "1" ]]; then
  git pull --ff-only
fi

ARGS=(
  main_batch_run.py
  --batch_config "$BATCH_CONFIG"
  --status_path "$STATUS_PATH"
)

if [[ "$ONLY_FAILED" == "1" ]]; then
  ARGS+=(--only_failed)
fi

if [[ "$RUN_BATCH" == "1" ]]; then
  ARGS+=(--run)
else
  ARGS+=(--dry_run)
fi

"$PYTHON_BIN" "${ARGS[@]}"

echo "Week7 batch workflow finished. Registry: $STATUS_PATH"
