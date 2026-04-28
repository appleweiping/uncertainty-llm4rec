# Server Runbook

This is a server execution framework for future Storyflow / TRUCE-Rec work. It
does not record any completed server run.

## Status

No server experiment, Qwen3-8B inference, Qwen3-8B + LoRA training, large-scale
baseline, or full Amazon run has been executed by Codex in this repository.

## Server Responsibilities

Server hardware is expected for:

- Qwen3-8B full inference when local execution is too heavy;
- Qwen3-8B + LoRA training;
- large Amazon Reviews 2023 category preprocessing;
- large-scale SASRec, BERT4Rec, GRU4Rec, LightGCN, and generative baseline
  runs where feasible;
- long-running observation, simulation, and triage experiments.

Local execution remains responsible for repository editing, small tests,
documentation, small real-data sanity checks, and report generation.

## Expected Server Directory Layout

Future server scripts and configs should use:

```text
configs/server/
scripts/server/
outputs/
runs/
data/raw/
data/cache/
```

Raw data, caches, outputs, and runs must not be committed unless a future task
explicitly adds sanitized lightweight fixtures.

## Environment Requirements

Future server setup should record:

- git remote and commit hash;
- Python version and package lock or exact install commands;
- CUDA, driver, GPU type, and GPU memory;
- model source and revision;
- dataset source, local path, checksum where available, and license/access
  notes;
- seed values and deterministic settings where feasible.

API keys and tokens must be supplied through environment variables or server
secret management. They must not be written into source files, configs, logs, or
commits.

## Planned Server Workflow

Future runbooks should expand these placeholders into exact commands after the
scripts exist:

1. Clone or update the repository on the server.
2. Verify the repository is on `main` and at the intended commit.
3. Create or activate the Python environment.
4. Install dependencies from the project dependency file.
5. Place or download datasets according to the dataset manifest.
6. Run preprocessing with manifest output.
7. Run Qwen3-8B inference or Qwen3-8B + LoRA training with config snapshots.
8. Write logs, metrics, generated outputs, and run manifests under `runs/` or
   `outputs/`.
9. Copy only sanitized summaries, configs, logs, and metrics needed for local
   analysis.

## Amazon Reviews 2023 Beauty Full Run Gate

Amazon Beauty full download and preprocessing should run on a server or a local
machine with large enough disk/network capacity. Codex has not run a full
Amazon Beauty download or preprocessing job.

Required inputs:

- repository checkout on `main`;
- `configs/datasets/amazon_reviews_2023_beauty.yaml`;
- access to `McAuley-Lab/Amazon-Reviews-2023` according to its dataset card and
  usage terms;
- raw review JSONL at
  `data/raw/amazon_reviews_2023_beauty/raw_review_All_Beauty.jsonl`;
- raw metadata JSONL at
  `data/raw/amazon_reviews_2023_beauty/raw_meta_All_Beauty.jsonl`;
- enough storage for raw/cache/processed outputs.

Lightweight readiness check:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run
```

Optional online availability check:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --check-online
```

Prepare dry-run/readiness:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run
```

Full prepare command shape after raw JSONL placement:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --reviews-jsonl data/raw/amazon_reviews_2023_beauty/raw_review_All_Beauty.jsonl --metadata-jsonl data/raw/amazon_reviews_2023_beauty/raw_meta_All_Beauty.jsonl --output-suffix full
```

Expected processed outputs:

- `item_catalog.csv`
- `interactions.csv`
- `user_sequences.jsonl`
- `observation_examples.jsonl`
- `preprocess_manifest.json`

Every server full run must preserve:

- exact command line;
- git commit hash;
- dataset config snapshot;
- raw file paths and checksums where possible;
- stdout/stderr logs;
- processed output manifest;
- failure/resume notes.

Only sanitized manifests, logs, and metrics should be copied back for local
analysis. Raw data and full processed data remain uncommitted.

## Required Run Artifacts

Every future server run should produce:

- command line;
- git commit hash;
- config snapshot;
- dataset manifest;
- environment summary;
- seed values;
- stdout/stderr log path;
- output JSONL or metric path;
- failure/resume status;
- notes on whether the run is synthetic, pilot, or full.

## Claim Policy

Do not state that a server run succeeded unless its log or artifact is present
and inspected. Do not convert planned commands into experimental results.
