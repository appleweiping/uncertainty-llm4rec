#!/bin/bash
# Download and preprocess Amazon Reviews 2023 data for TRUCE-Rec
# Runs on server. Downloads from HuggingFace, preprocesses, removes raw to save disk.
set -euo pipefail

PROJECT_DIR="$HOME/projects/TRUCE-Rec"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate qwen_vllm

export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# Domain configs
declare -A DOMAIN_CATEGORIES=(
    ["beauty"]="All_Beauty"
    ["books"]="Books"
    ["electronics"]="Electronics"
    ["movies"]="Movies_and_TV"
    ["sports"]="Sports_and_Outdoors"
    ["toys"]="Toys_and_Games"
)

DOMAIN=${1:-"all"}

download_and_prepare_domain() {
    local domain=$1
    local category=${DOMAIN_CATEGORIES[$domain]}
    local raw_dir="data/raw/amazon_reviews_2023_${domain}"
    local reviews_file="${raw_dir}/${category}.jsonl"
    local meta_file="${raw_dir}/meta_${category}.jsonl"

    echo ""
    echo "=========================================="
    echo "Processing domain: $domain ($category)"
    echo "=========================================="

    # Check if already processed
    local processed_dir="data/processed/amazon_reviews_2023_${domain}"
    if [ -f "${processed_dir}/full/test.jsonl" ]; then
        echo "  Already processed, skipping."
        return 0
    fi

    mkdir -p "$raw_dir"

    # Download reviews
    if [ ! -f "$reviews_file" ] && [ ! -f "${reviews_file}.gz" ]; then
        echo "  Downloading reviews..."
        python -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id='McAuley-Lab/Amazon-Reviews-2023',
    filename='raw/review_categories/${category}.jsonl.gz',
    repo_type='dataset',
    local_dir='${raw_dir}',
    local_dir_use_symlinks=False,
)
print(f'  Downloaded: {path}')
" || {
            echo "  ERROR: Failed to download reviews for $domain"
            return 1
        }
        # Move file to expected location if needed
        if [ -f "${raw_dir}/raw/review_categories/${category}.jsonl.gz" ]; then
            mv "${raw_dir}/raw/review_categories/${category}.jsonl.gz" "${reviews_file}.gz"
            rm -rf "${raw_dir}/raw"
        fi
    fi

    # Download metadata
    if [ ! -f "$meta_file" ] && [ ! -f "${meta_file}.gz" ]; then
        echo "  Downloading metadata..."
        python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='McAuley-Lab/Amazon-Reviews-2023',
    filename='raw/meta_categories/meta_${category}.jsonl.gz',
    repo_type='dataset',
    local_dir='${raw_dir}',
    local_dir_use_symlinks=False,
)
print(f'  Downloaded: {path}')
" || {
            echo "  ERROR: Failed to download metadata for $domain"
            return 1
        }
        if [ -f "${raw_dir}/raw/meta_categories/meta_${category}.jsonl.gz" ]; then
            mv "${raw_dir}/raw/meta_categories/meta_${category}.jsonl.gz" "${meta_file}.gz"
            rm -rf "${raw_dir}/raw"
        fi
    fi

    # Decompress if needed
    if [ -f "${reviews_file}.gz" ] && [ ! -f "$reviews_file" ]; then
        echo "  Decompressing reviews..."
        gunzip -k "${reviews_file}.gz"
    fi
    if [ -f "${meta_file}.gz" ] && [ ! -f "$meta_file" ]; then
        echo "  Decompressing metadata..."
        gunzip -k "${meta_file}.gz"
    fi

    # Preprocess
    echo "  Preprocessing..."
    python scripts/prepare_amazon_reviews_2023.py \
        --dataset "amazon_reviews_2023_${domain}" \
        --reviews-jsonl "$reviews_file" \
        --metadata-jsonl "$meta_file" \
        --output-suffix full \
        --allow-full

    # Verify
    if [ -f "${processed_dir}/full/test.jsonl" ]; then
        local test_count=$(wc -l < "${processed_dir}/full/test.jsonl")
        echo "  SUCCESS: $domain processed. Test set: $test_count examples."
    else
        echo "  ERROR: Processing failed for $domain"
        return 1
    fi

    # Clean up raw to save disk (keep .gz for reproducibility)
    echo "  Cleaning raw JSONL (keeping .gz)..."
    rm -f "$reviews_file" "$meta_file"

    echo "  Done: $domain"
}

if [ "$DOMAIN" = "all" ]; then
    for d in beauty books electronics movies sports toys; do
        download_and_prepare_domain "$d"
    done
else
    download_and_prepare_domain "$DOMAIN"
fi

echo ""
echo "=== Data preparation complete ==="
echo "Processed domains:"
for d in beauty books electronics movies sports toys; do
    processed="data/processed/amazon_reviews_2023_${d}/full/test.jsonl"
    if [ -f "$processed" ]; then
        count=$(wc -l < "$processed")
        echo "  $d: $count test examples"
    else
        echo "  $d: NOT READY"
    fi
done
