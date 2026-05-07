# Server Execution Matrix

This document is the server-side checklist for moving from current pilots to a
complete four-domain experiment suite. It contains commands and acceptance
criteria, not results.

## Domains

The target full-scale domains are:

| Domain | Week8 source | Intended TRUCE output |
| --- | --- | --- |
| beauty | existing TRUCE/Beauty pipeline first, Week8 if available | `data/processed/week8_same_candidate/beauty_large10000_100neg/{valid,test}` |
| books | Week8 external task | `data/processed/week8_same_candidate/books_large10000_100neg/{valid,test}` |
| electronics | Week8 external task | `data/processed/week8_same_candidate/electronics_large10000_100neg/{valid,test}` |
| movies | Week8 external task | `data/processed/week8_same_candidate/movies_large10000_100neg/{valid,test}` |

Do not resample users, negatives, histories, or candidates.

## Pull And Status

```bash
cd ~/projects/TRUCE-Rec
git pull --ff-only
source .venv_truce/bin/activate

python scripts/summarize_controlled_baseline_suite.py
```

## Week8 File Check

```bash
find ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  -path "*large10000_100neg*" -type f | sort
```

Expected per split/domain:

- `ranking_valid.jsonl` or `ranking_test.jsonl`;
- `candidate_items.csv`;
- `train_interactions.csv`;
- `item_metadata.csv`;
- `selected_users.csv`;
- `metadata.json`.

## Convert Week8 Tasks

Preferred: generate commands locally or on the server:

```bash
python scripts/plan_four_domain_server_runs.py \
  --source-root ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  --output-root data/processed/week8_same_candidate \
  --domains beauty books electronics movies \
  --splits valid test
```

Then run the printed commands when the corresponding task directories exist.

Manual example:

```bash
python scripts/convert_week8_same_candidate_to_truce.py \
  --task-dir ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate \
  --output-dir data/processed/week8_same_candidate/books_large10000_100neg/test \
  --domain books \
  --split test
```

Acceptance:

- `preprocess_manifest.json` exists;
- `example_count > 0`;
- each event has target included in candidates;
- `candidate_row_count` equals the sum of all candidate rows;
- event/source IDs are preserved in example metadata.

## Generate External Packets

For the current Beauty packet lane:

```bash
python scripts/prepare_controlled_baseline_suite.py
```

For the added official candidates:

```bash
python scripts/prepare_controlled_baseline_suite.py \
  --suite-name qwen3_base_adapter_main6_amazon_beauty \
  --include-added-official-candidates
```

For Week8 domains, add domain-specific packet configs before running official
baselines. Do not reuse Beauty packet configs for books/electronics/movies.

## Run Baselines

Priority order:

1. Cheap traditional/retrieval/sequential baselines.
2. Current controlled-adapter pilots only as pipeline diagnostics.
3. Official-native controlled baselines after fidelity audit.
4. Ours full.
5. Ours ablations.

Every method must end with:

```text
candidate_scores.csv -> predictions.jsonl -> metrics.json + metrics.csv
```

## Import And Evaluate

For completed controlled baseline scores:

```bash
python scripts/import_evaluate_controlled_baseline.py \
  --manifest outputs/server_training/controlled_baselines/<name>/controlled_baseline_manifest.json
```

Then:

```bash
python scripts/summarize_controlled_baseline_suite.py
```

Paper eligibility requires:

- `implementation_fidelity=official_native_controlled` for official main
  baselines;
- `official_algorithm_reused=true`;
- TRUCE `metrics.json`;
- raw score artifacts and logs retained.

## Observation And Framework Analysis

After predictions are available:

```bash
python scripts/analyze_observation.py --run-dir <observation_or_baseline_run_dir>
```

For CURE/TRUCE feature and reranking analysis, follow:

- `scripts/build_confidence_features.py`;
- `scripts/calibrate_confidence_features.py`;
- `scripts/residualize_confidence_features.py`;
- `scripts/rerank_confidence_features.py`;
- `scripts/triage_confidence_features.py`.

Only use train/valid fit splits for learned calibration/residualization. Same
split fitting is diagnostic only.

## Final Artifact Gate

Before paper table export, each run needs:

- resolved config or manifest;
- environment and git info;
- stdout/stderr logs;
- raw scores or raw LLM responses where applicable;
- `predictions.jsonl`;
- `metrics.json`;
- `metrics.csv`;
- cost/latency or runtime summary;
- official-fidelity audit for official baselines.
