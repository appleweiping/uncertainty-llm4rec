# Amazon Reviews 2023 Readiness

Amazon Reviews 2023 is the intended full-scale e-commerce data source after
MovieLens sanity checks. This document defines the entry point; it does not
claim that full data has been downloaded or processed.

## Beauty Config

Primary config:

```text
configs/datasets/amazon_reviews_2023_beauty.yaml
```

The config records:

- source dataset and category;
- local raw/cache/processed paths;
- expected review and metadata fields;
- title/user/item/rating/timestamp fields;
- metadata join plan;
- k-core and interaction filtering settings;
- local sample mode;
- server full mode command template.

## Lightweight Inspect

Default dry-run/readiness command:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run
```

This writes ignored outputs under:

```text
outputs/amazon_reviews_2023/amazon_reviews_2023_beauty/inspect/
```

It does not download full data and does not claim availability success. Optional
online checks must be requested explicitly:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --check-online
```

If network, license, login, or dependency conditions fail, the script records a
clear status and recovery command instead of silently skipping.

## Prepare Entry

Dry-run/readiness:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run
```

Full prepare command shape once raw JSONL files exist:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --reviews-jsonl data/raw/amazon_reviews_2023_beauty/raw_review_All_Beauty.jsonl --metadata-jsonl data/raw/amazon_reviews_2023_beauty/raw_meta_All_Beauty.jsonl --output-suffix full
```

The prepare code defines:

- review row -> canonical interaction conversion;
- metadata row -> item catalog conversion;
- title cleaning;
- timestamp sorting;
- interaction-count filtering;
- k-core filtering;
- leave-last-two and rolling/global chronological settings;
- item popularity and head/mid/tail bucket assignment.

## Server Guidance

Full Amazon Beauty download and preprocessing should run on a server or local
machine with enough disk, network, and runtime budget. Codex has not run this
full pipeline and cannot claim server completion without logs/artifacts.
