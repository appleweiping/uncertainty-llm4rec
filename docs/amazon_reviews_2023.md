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

Current local raw Beauty files detected by config:

```text
data/raw/amazon_reviews_2023_beauty/All_Beauty.jsonl
data/raw/amazon_reviews_2023_beauty/meta_All_Beauty.jsonl
```

Compressed copies may also exist next to them. Raw and processed data remain
ignored by git.

Additional local raw Amazon category configs currently registered:

- `configs/datasets/amazon_reviews_2023_digital_music.yaml`
- `configs/datasets/amazon_reviews_2023_handmade.yaml`
- `configs/datasets/amazon_reviews_2023_health.yaml`

Beauty remains the first full-data gate; the extra categories are robustness
candidates after the Beauty path is stable.

## Lightweight Inspect

Default dry-run/readiness command:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run --sample-records 3
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

Full prepare command shape once raw JSONL files exist and the full run is
explicitly approved:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --reviews-jsonl data/raw/amazon_reviews_2023_beauty/All_Beauty.jsonl --metadata-jsonl data/raw/amazon_reviews_2023_beauty/meta_All_Beauty.jsonl --output-suffix full --allow-full
```

Small sample prepare command shape for local pipeline sanity:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --sample-mode --max-records 5000 --output-suffix sample_5k --min-user-interactions 1 --user-k-core 1 --item-k-core 1 --min-history 1 --max-history 20
```

Sample prepare only scans the requested review prefix and then loads metadata
for the surviving item ids. It writes `is_sample_result=true`,
`is_full_result=false`, and `is_experiment_result=false` in the preprocess
manifest. It is not a full processed result and must not be used for paper
claims.

The script blocks accidental full preprocessing unless `--allow-full` is
provided. This prevents a local readiness command from becoming an unapproved
full-data run.

The prepare code defines:

- review row -> canonical interaction conversion;
- metadata row -> item catalog conversion;
- title cleaning;
- timestamp sorting;
- interaction-count filtering;
- k-core filtering;
- leave-last-two and rolling/global chronological settings;
- item popularity and head/mid/tail bucket assignment.

## Sample Observation Gate

After the local `sample_5k` prepare exists, validate it before constructing any
prompt inputs:

```powershell
python scripts/validate_processed_dataset.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k
```

Then build a small stratified input file for future observation:

```powershell
python scripts/build_observation_inputs.py --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --split test --max-examples 30 --stratify-by-popularity
```

Expected ignored output:

```text
outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl
```

This gate only proves that the processed sample can feed the title-level
generative observation pipeline. It does not call an API, does not train a
model, and does not convert the sample into a full Amazon result.

## Server Guidance

Full Amazon Beauty download and preprocessing should run on a server or local
machine with enough disk, network, and runtime budget. Codex has not run this
full pipeline and cannot claim server completion without logs/artifacts.
