# Server Runbook

This is a server execution framework for future Storyflow / TRUCE-Rec work. It
does not record any completed server run.

## Status

No server experiment, Qwen3-8B inference, Qwen3-8B + LoRA training, or
large-scale baseline has been executed by Codex in this repository. Amazon
Beauty full preprocessing has been executed locally as a data-readiness
artifact. User-approved DeepSeek local API full-slice diagnostics have been
executed on Amazon Beauty repeat-free inputs; these are local API artifacts
under ignored `outputs/`, not server runs and not paper conclusions.

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

Before any real server, full-data, API, or trained-baseline expansion, generate
the local approval checklist:

```powershell
python scripts/build_expansion_approval_checklist.py
```

The command writes ignored artifacts under
`outputs/approval_gates/next_expansion/`. It does not execute the server, call
an API, train a model, download data, or authorize a run by itself.

After selecting one expansion track, generate a non-executing run packet:

```powershell
python scripts/build_expansion_run_packet.py --track qwen3_server --run-label qwen3_beauty_plan_packet --input-jsonl <input-jsonl> --target-output-dir <server-output-dir>
```

The packet writes ignored artifacts under `outputs/run_packets/` by default. It
lists missing confirmations, safe preflight commands, approval-required command
shape, expected artifacts, and forbidden claims. It still does not execute a
server command, call an API, process full data, train a model, or authorize
execution by itself.

## Qwen3-8B Observation Scaffold

The repository includes a server observation contract for Qwen3-8B:

- config: `configs/server/qwen3_8b_observation.yaml`;
- script: `scripts/server/run_qwen3_observation.py`;
- package helper: `storyflow.server`;
- default mode: plan-only, with no model loading and no inference;
- guarded execution mode: `--execute-server`, intended only for approved server
  hardware.

Plan-only command from the repository root:

```powershell
python scripts/server/run_qwen3_observation.py --config configs/server/qwen3_8b_observation.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl --output-dir outputs/server_observations/qwen3_8b/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_plan --max-examples 20 --run-label qwen3_beauty_plan
```

The plan writes ignored artifacts:

- `request_records.jsonl`;
- `expected_output_contract.json`;
- `server_command_plan.md`;
- `manifest.json`.

The plan manifest must contain:

- `api_called=false`;
- `server_executed=false`;
- `model_inference_run=false`;
- `model_training=false`;
- `output_schema_matches_api_observation=true`;
- `grounding_required_before_correctness=true`;
- `is_experiment_result=false`.

Approved server execution command shape:

```powershell
python scripts/server/run_qwen3_observation.py --config configs/server/qwen3_8b_observation.yaml --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json.jsonl --output-dir outputs/server_observations/qwen3_8b/amazon_reviews_2023_beauty/full/test_no_repeat_forced_json_server --execute-server --run-stage full --run-label qwen3_beauty_full_server
```

This command requires a server Python environment with `torch`, `transformers`,
the configured Qwen3 model source/cache path, sufficient GPU memory, the input
JSONL, and the matching processed catalog CSV. It writes API-compatible layers:

- `request_records.jsonl`;
- `raw_responses.jsonl`;
- `parsed_predictions.jsonl`;
- `failed_cases.jsonl`;
- `grounded_predictions.jsonl`;
- `metrics.json`;
- `report.md`;
- `manifest.json`.

Codex must not claim this server command has run unless the user provides the
server manifest and logs. Qwen3 outputs must still be grounded to the catalog
before correctness, confidence, popularity, or head/mid/tail metrics are
reported.

## Amazon Reviews 2023 Beauty Full Run Gate

Amazon Beauty full download and preprocessing should run on a server or a local
machine with large enough disk/network capacity. A local full preprocessing job
has been run on 2026-04-29 from already placed raw JSONL files; this is not a
server run and not an experiment result.

Required inputs:

- repository checkout on `main`;
- `configs/datasets/amazon_reviews_2023_beauty.yaml`;
- access to `McAuley-Lab/Amazon-Reviews-2023` according to its dataset card and
  usage terms;
- raw review JSONL at
  `data/raw/amazon_reviews_2023_beauty/All_Beauty.jsonl`;
- raw metadata JSONL at
  `data/raw/amazon_reviews_2023_beauty/meta_All_Beauty.jsonl`;
- enough storage for raw/cache/processed outputs.

Lightweight readiness check:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run --sample-records 3
```

Optional online availability check:

```powershell
python scripts/inspect_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --check-online
```

Prepare dry-run/readiness:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --dry-run
```

Full prepare command shape after raw JSONL placement and explicit full-run
approval:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --reviews-jsonl data/raw/amazon_reviews_2023_beauty/All_Beauty.jsonl --metadata-jsonl data/raw/amazon_reviews_2023_beauty/meta_All_Beauty.jsonl --output-suffix full --allow-full
```

Local sample prepare before full run:

```powershell
python scripts/prepare_amazon_reviews_2023.py --dataset amazon_reviews_2023_beauty --sample-mode --max-records 5000 --output-suffix sample_5k --min-user-interactions 1 --user-k-core 1 --item-k-core 1 --min-history 1 --max-history 20
```

Sample prepare is only a pipeline readiness check. It is not a full server run
and must not be reported as experimental evidence. The prepare script blocks
accidental full preprocessing unless `--allow-full` is passed.

Expected processed outputs:

- `item_catalog.csv`
- `interactions.csv`
- `item_popularity.csv`
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

## Amazon Cross-Category Readiness Matrix

Before adding another Amazon full run, inspect configured category readiness
without downloading data:

```powershell
python scripts/inspect_amazon_category_matrix.py --sample-records 3
```

This writes ignored JSON/CSV/Markdown artifacts under
`outputs/amazon_reviews_2023/category_matrix/`. It records whether raw review
and metadata JSONL files exist for Beauty, Digital_Music, Handmade_Products,
Health_and_Personal_Care, Video_Games, Sports_and_Outdoors, and Books. It also
records guarded sample/full command templates and marks `api_called=false`,
`server_executed=false`, `full_download_attempted=false`, and
`is_experiment_result=false`.

Video_Games and Books are the preferred title-rich robustness candidates after
Beauty. Their full preparation remains server/manual-raw gated and must not be
claimed until raw files, commands, logs, manifests, and validation artifacts are
available.

## Baseline Ranking Run Manifest Contract

Large SASRec, BERT4Rec, GRU4Rec, LightGCN, and similar baseline runs must write
a source run manifest before their ranking JSONL is adapted through
`ranking_jsonl`. A safe template is committed at
`configs/server/baseline_ranking_run_manifest.example.json`.

Required validation command after copying the manifest and declared safe logs
or ranking paths back to the local repository:

```powershell
python scripts/validate_baseline_run_manifest.py --manifest-json runs/baselines/sasrec/amazon_reviews_2023_beauty/full/run_manifest.json --strict
```

The validator records an ignored manifest under
`outputs/baseline_run_manifest_validation/...`. It checks required provenance,
train/evaluation split separation, command/git/seed metadata, path existence
and hashes when available, and the leakage guards
`grounding_required_before_correctness=true` and
`uses_heldout_targets_for_training=false`. It does not execute the baseline,
call APIs, train models, download data, or create a paper result. The ranking
JSONL must still pass `scripts/validate_baseline_artifact.py` before grounded
title-level observation.

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
