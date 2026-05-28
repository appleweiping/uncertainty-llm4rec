#!/bin/bash
# TRUCE-Rec Server Deployment Script
# Target: pony-rec-gpu (125.71.97.70:15302, user ajifang)
# Conda env: qwen_vllm (Python 3.12, PyTorch 2.8, CUDA 12.8)
set -euo pipefail

PROJECT_DIR="$HOME/projects/TRUCE-Rec"
CONDA_ENV="qwen_vllm"
MODEL_PATH="$HOME/models/Qwen/Qwen3-8B"
GITHUB_REPO="https://github.com/appleweiping/TRUCE-Rec.git"

echo "=== TRUCE-Rec Server Deployment ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# 1. Clone or update repo
if [ -d "$PROJECT_DIR" ]; then
    echo "[1/5] Updating existing repo..."
    cd "$PROJECT_DIR"
    git fetch origin
    git reset --hard origin/main
else
    echo "[1/5] Cloning repo..."
    git clone "$GITHUB_REPO" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# 2. Activate conda and install
echo "[2/5] Setting up Python environment..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
pip install -e . --quiet 2>/dev/null || pip install -e .

# 3. Verify model access
echo "[3/5] Verifying model access..."
if [ -d "$MODEL_PATH" ]; then
    echo "  Model found: $MODEL_PATH"
else
    echo "  ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# 4. Create data directories
echo "[4/5] Creating data directories..."
DOMAINS="beauty books electronics movies sports toys"
for domain in $DOMAINS; do
    mkdir -p "data/raw/amazon_reviews_2023_${domain}"
    mkdir -p "data/processed/amazon_reviews_2023_${domain}"
    mkdir -p "data/interim/amazon_reviews_2023_${domain}"
    mkdir -p "data/cache/amazon_reviews_2023_${domain}"
done
mkdir -p outputs/server_observations
mkdir -p outputs/baselines
mkdir -p outputs/ours_results
mkdir -p outputs/ablation

# 5. Verify setup
echo "[5/5] Verifying setup..."
python -c "
import sys
sys.path.insert(0, 'src')
import storyflow
import llm4rec
print(f'  storyflow: OK')
print(f'  llm4rec: OK')
print(f'  Python: {sys.version}')
"

echo ""
echo "=== Deployment Complete ==="
echo "Project: $PROJECT_DIR"
echo "Next steps:"
echo "  1. Download Amazon2023 data for 6 domains"
echo "  2. Run preprocessing pipeline"
echo "  3. Build observation inputs"
echo "  4. Run Qwen3-8B observation"
echo "  5. Run CU-GR policy + evaluation"
