#!/bin/bash
# TRUCE-Rec Full Experiment Pipeline
# Runs on server: pony-rec-gpu
# Usage: bash scripts/server/run_full_pipeline.sh [domain]
set -euo pipefail

PROJECT_DIR="$HOME/projects/TRUCE-Rec"
cd "$PROJECT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm
export PYTHONPATH="$PROJECT_DIR/src"

MODEL_PATH="$HOME/models/Qwen/Qwen3-8B"
DOMAIN=${1:-"beauty"}

echo "=== TRUCE-Rec Full Pipeline: $DOMAIN ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Paths
PONY_EXTERNAL="$HOME/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks"
OLD_PROCESSED="$HOME/projects/uncertainty-llm4rec/data/processed"

# Step 1: Convert Pony data to TRUCE format
echo "[Step 1/5] Converting data to TRUCE format..."
CONVERTED_DIR="data/processed/amazon_reviews_2023_${DOMAIN}/week8_converted"

if [ -f "${CONVERTED_DIR}/examples.jsonl" ]; then
    echo "  Already converted, skipping."
else
    # Determine source directory
    if [ "$DOMAIN" = "beauty" ]; then
        TASK_DIR="${PONY_EXTERNAL}/beauty_same_candidate"
        # Beauty uses the old uncertainty-llm4rec ranking file
        if [ ! -f "${TASK_DIR}/ranking_test.jsonl" ]; then
            cp "${OLD_PROCESSED}/amazon_beauty/ranking_test.jsonl" "${TASK_DIR}/ranking_test.jsonl"
        fi
    else
        TASK_DIR="${PONY_EXTERNAL}/${DOMAIN}_large10000_100neg_test_same_candidate"
    fi

    if [ ! -d "$TASK_DIR" ]; then
        echo "  ERROR: Task directory not found: $TASK_DIR"
        echo "  This domain may need fresh data preparation."
        exit 1
    fi

    python scripts/convert_week8_same_candidate_to_truce.py \
        --task-dir "$TASK_DIR" \
        --output-dir "$CONVERTED_DIR" \
        --domain "$DOMAIN" \
        --split test

    echo "  Converted: $(wc -l < ${CONVERTED_DIR}/examples.jsonl) examples"
fi

# Step 2: Build observation inputs (prompts for Qwen3-8B)
echo "[Step 2/5] Building observation inputs..."
OBS_INPUT="outputs/observation_inputs/${DOMAIN}_test_forced_json.jsonl"

if [ -f "$OBS_INPUT" ]; then
    echo "  Already built, skipping."
else
    mkdir -p outputs/observation_inputs
    python scripts/build_week8_observation_inputs.py \
        --processed-dir "$CONVERTED_DIR" \
        --dataset "amazon_reviews_2023_${DOMAIN}" \
        --domain "$DOMAIN" \
        --split test \
        --prompt-template forced_json \
        --output-jsonl "$OBS_INPUT"

    echo "  Built: $(wc -l < $OBS_INPUT) observation inputs"
fi

# Step 3: Run Qwen3-8B observation (GPU inference)
echo "[Step 3/5] Running Qwen3-8B observation..."
OBS_OUTPUT="outputs/server_observations/qwen3_8b/${DOMAIN}"

if [ -f "${OBS_OUTPUT}/manifest.json" ]; then
    echo "  Already completed, skipping."
else
    mkdir -p "$OBS_OUTPUT"
    python scripts/server/run_qwen3_observation.py \
        --config configs/server/qwen3_8b_observation.yaml \
        --input-jsonl "$OBS_INPUT" \
        --output-dir "$OBS_OUTPUT" \
        --execute-server

    echo "  Observation complete: $(wc -l < ${OBS_OUTPUT}/grounded_predictions.jsonl) predictions"
fi

# Step 4: Build confidence features + calibrate
echo "[Step 4/5] Building confidence features and calibrating..."
FEATURES_DIR="outputs/confidence_features/${DOMAIN}"
mkdir -p "$FEATURES_DIR"

if [ -f "${FEATURES_DIR}/calibrated_features.jsonl" ]; then
    echo "  Already calibrated, skipping."
else
    python scripts/build_confidence_features.py \
        --observation-dir "$OBS_OUTPUT" \
        --output-dir "$FEATURES_DIR" \
        --domain "$DOMAIN"

    python scripts/calibrate_confidence_features.py \
        --features-dir "$FEATURES_DIR" \
        --domain "$DOMAIN"

    echo "  Features built and calibrated."
fi

# Step 5: Run CU-GR policy + evaluate
echo "[Step 5/5] Running CU-GR policy and evaluation..."
RESULTS_DIR="outputs/ours_results/${DOMAIN}"
mkdir -p "$RESULTS_DIR"

python scripts/rerank_confidence_features.py \
    --features-dir "$FEATURES_DIR" \
    --output-dir "$RESULTS_DIR" \
    --domain "$DOMAIN"

python scripts/evaluate_predictions.py \
    --predictions-dir "$RESULTS_DIR" \
    --domain "$DOMAIN" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=== Pipeline Complete: $DOMAIN ==="
echo "Results: $RESULTS_DIR"
if [ -f "${RESULTS_DIR}/metrics.json" ]; then
    echo "Metrics:"
    python -c "import json; m=json.load(open('${RESULTS_DIR}/metrics.json')); print(f'  HR@10: {m.get(\"HR@10\", \"N/A\")}'); print(f'  NDCG@10: {m.get(\"NDCG@10\", \"N/A\")}'); print(f'  MRR: {m.get(\"MRR\", \"N/A\")}')"
fi
