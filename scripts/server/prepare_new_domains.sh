#!/bin/bash
# Prepare sports and toys domains from existing raw .gz files
# Uses the old uncertainty-llm4rec raw data, processes through TRUCE-Rec pipeline
set -euo pipefail

PROJECT_DIR="$HOME/projects/TRUCE-Rec"
OLD_RAW="$HOME/projects/uncertainty-llm4rec/data/raw"
cd "$PROJECT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm
export PYTHONPATH="$PROJECT_DIR/src"

prepare_domain() {
    local domain=$1
    local category=$2
    local raw_source="${OLD_RAW}/amazon_${domain}"
    local truce_raw="data/raw/amazon_reviews_2023_${domain}"
    local processed="data/processed/amazon_reviews_2023_${domain}/full"

    echo ""
    echo "=========================================="
    echo "Preparing domain: $domain ($category)"
    echo "=========================================="

    # Check if already processed
    if [ -f "${processed}/ranking_test.jsonl" ]; then
        echo "  Already processed, skipping."
        return 0
    fi

    # Symlink raw .gz files
    mkdir -p "$truce_raw"
    if [ ! -f "${truce_raw}/${category}.jsonl.gz" ]; then
        ln -sf "${raw_source}/${category}.jsonl.gz" "${truce_raw}/${category}.jsonl.gz"
    fi
    if [ ! -f "${truce_raw}/meta_${category}.jsonl.gz" ]; then
        ln -sf "${raw_source}/meta_${category}.jsonl.gz" "${truce_raw}/meta_${category}.jsonl.gz"
    fi

    # Decompress (one at a time to save disk)
    echo "  Decompressing reviews..."
    if [ ! -f "${truce_raw}/${category}.jsonl" ]; then
        gunzip -k "${truce_raw}/${category}.jsonl.gz"
    fi
    echo "  Decompressing metadata..."
    if [ ! -f "${truce_raw}/meta_${category}.jsonl" ]; then
        gunzip -k "${truce_raw}/meta_${category}.jsonl.gz"
    fi

    # Run TRUCE-Rec preprocessing
    echo "  Running preprocessing pipeline..."
    python scripts/prepare_amazon_reviews_2023.py \
        --dataset "amazon_reviews_2023_${domain}" \
        --reviews-jsonl "${truce_raw}/${category}.jsonl" \
        --metadata-jsonl "${truce_raw}/meta_${category}.jsonl" \
        --output-suffix full \
        --allow-full

    # Verify
    if [ -f "${processed}/ranking_test.jsonl" ]; then
        local count=$(wc -l < "${processed}/ranking_test.jsonl")
        echo "  SUCCESS: $count test examples"
    else
        echo "  ERROR: preprocessing failed"
        # Clean up decompressed files
        rm -f "${truce_raw}/${category}.jsonl" "${truce_raw}/meta_${category}.jsonl"
        return 1
    fi

    # Clean up decompressed files to save disk
    echo "  Cleaning up decompressed files..."
    rm -f "${truce_raw}/${category}.jsonl" "${truce_raw}/meta_${category}.jsonl"

    echo "  Done: $domain"
}

# Process sports and toys
prepare_domain "sports" "Sports_and_Outdoors"
prepare_domain "toys" "Toys_and_Games"

echo ""
echo "=== Data preparation complete ==="
