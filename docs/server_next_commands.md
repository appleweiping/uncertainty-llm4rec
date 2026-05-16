# Server Next Commands

This is the short handoff command sheet after switching TRUCE paper-facing
external baselines to reused Pony/Uncertainty official-qwen3base same-candidate
evidence.

## Pull Latest

```bash
cd ~/projects/TRUCE-Rec
git pull --ff-only
```

## Recommended Server Entrypoints

For Week8 four-domain conversion plus Ours adapter prep:

```bash
cd ~/projects/TRUCE-Rec
bash scripts/server/run_week8_four_domain_pipeline.sh
```

Import/copy Pony official baseline evidence locally:

```powershell
py -3 scripts\import_pony_official_baselines.py `
  --pony-root D:\Research\Uncertainty `
  --output-root outputs\pony_official_baselines `
  --manifest configs\baselines\pony_official_external_baselines.yaml
```

Then build the TRUCE-side comparison/status tables:

```powershell
py -3 scripts\build_pony_baseline_comparison.py `
  --manifest-json outputs\pony_official_baselines\manifest.json `
  --output-root outputs\pony_official_baselines\tables `
  --output-name pony_official_baseline_comparison
```

The old `scripts/server/run_controlled_baseline_queue.sh` lane is legacy/pilot
only. Do not run it as the default paper-baseline path.

## Import And Evaluate

For legacy controlled-adapter pilot diagnostics only, after a full run
completes:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

for NAME in \
  tallrec_qwen3_lora_amazon_beauty \
  dealrec_qwen3_lora_amazon_beauty \
  lc_rec_qwen3_lora_amazon_beauty
do
  case "$NAME" in
    tallrec_qwen3_lora_amazon_beauty) PROJECT=tallrec ;;
    dealrec_qwen3_lora_amazon_beauty) PROJECT=dealrec ;;
    lc_rec_qwen3_lora_amazon_beauty) PROJECT=lc_rec ;;
  esac
  BASE=outputs/server_training/controlled_baselines/$NAME
  RUN=outputs/runs/${NAME}_seed13
  mkdir -p "$RUN/artifacts"
  cp "$BASE/candidate_scores.csv" "$RUN/artifacts/candidate_scores.csv"
  python scripts/import_external_predictions.py \
    --scores "$RUN/artifacts/candidate_scores.csv" \
    --examples "outputs/server_packets/${PROJECT}_amazon_beauty/truce_examples.jsonl" \
    --output "$RUN/predictions.jsonl" \
    --method "$NAME" \
    --source-project "$PROJECT" \
    --model-name Qwen3-8B-LoRA \
    --seed 13 \
    --split test
  python scripts/evaluate_predictions.py \
    --predictions "$RUN/predictions.jsonl" \
    --output-dir "$RUN"
done
```

If shell prefix expansion is confusing, run each baseline manually with the
packet paths:

- TALLRec packet:
  `outputs/server_packets/tallrec_amazon_beauty/truce_examples.jsonl`
- DEALRec packet:
  `outputs/server_packets/dealrec_amazon_beauty/truce_examples.jsonl`
- LC-Rec packet:
  `outputs/server_packets/lc_rec_amazon_beauty/truce_examples.jsonl`

## Evidence Rule

Pony official baseline rows are paper-facing only when the manifest marks them
`completed_result`, `same_schema_external_baseline`, `official_completed`, and
the local copied evidence tarball is present. Smoke metrics and legacy
controlled-adapter outputs are not paper results. Ours may tune hyperparameters
only on the declared validation protocol.

## Status Summary

Use `outputs/pony_official_baselines/manifest.json` and
`configs/baselines/pony_official_external_baselines.yaml` as the current
baseline status sources. Legacy controlled-baseline status summaries are
diagnostic only.

## Week8 Data Conversion

When the large same-candidate tasks are ready, convert one split/domain into a
TRUCE processed directory without resampling:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

python scripts/convert_week8_same_candidate_to_truce.py \
  --task-dir ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks/books_large10000_100neg_test_same_candidate \
  --output-dir data/processed/week8_same_candidate/books_large10000_100neg/test \
  --domain books \
  --split test \
  --strict-target-in-candidates
```

Repeat for the artifact slugs below, and preserve the original candidate/event
alignment. Do not edit `candidate_items.csv` or `ranking_valid/test.jsonl`.

```text
beauty=beauty_supplementary_smallerN_100neg
books=books_large10000_100neg
electronics=electronics_large10000_100neg
movies=movies_large10000_100neg
```

Generate all four-domain conversion commands:

```bash
python scripts/plan_four_domain_server_runs.py \
  --source-root ~/projects/pony-rec-rescue-shadow-v6/outputs/baselines/external_tasks \
  --output-root data/processed/week8_same_candidate \
  --domains beauty books electronics movies \
  --splits valid test
```

Validate converted artifacts:

```bash
python scripts/validate_week8_same_candidate_processed.py \
  --root data/processed/week8_same_candidate \
  --domains beauty books electronics movies \
  --splits valid test \
  --expected-candidates 101 \
  --expected-negatives 100
```

Do not add `--expected-users 10000` unless every selected artifact, including
Beauty, is expected to have exactly 10k users.

## Build And Run Week8 Observation Inputs

After conversion, build Qwen3 observation inputs. These are required for the
large-scale observation milestone: Beauty full-domain plus
books/electronics/movies 10k users, ideally matching the later formal training
and evaluation size.

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

for SPEC in \
  beauty:beauty_supplementary_smallerN_100neg \
  books:books_large10000_100neg \
  electronics:electronics_large10000_100neg \
  movies:movies_large10000_100neg
do
  DOMAIN=${SPEC%%:*}
  SLUG=${SPEC#*:}
  python scripts/build_week8_observation_inputs.py \
    --processed-dir data/processed/week8_same_candidate/${SLUG}/test \
    --dataset ${SLUG} \
    --domain ${DOMAIN} \
    --split test \
    --prompt-template forced_json
done
```

Base Qwen3-8B observation on one domain:

```bash
source ~/projects/TALLRec/.venv_tallrec/bin/activate

DOMAIN=books
SLUG=books_large10000_100neg
nohup python scripts/server/run_qwen3_observation.py \
  --input-jsonl outputs/observation_inputs/week8_same_candidate/${SLUG}/test_forced_json.jsonl \
  --output-dir outputs/server_observations/qwen3_8b/week8_same_candidate/${SLUG}/test_forced_json \
  --run-label qwen3_base_${DOMAIN}_week8_test \
  --run-stage observation \
  --execute-server \
  > outputs/logs/qwen3_base_observation_${DOMAIN}.log 2>&1 &
```

Observation is not complete until the same phenomenon analysis covers base
Qwen3-8B, Ours, and the reused strong baselines wherever the required
prediction/score artifacts are available. Legacy controlled-adapter outputs
remain pilots.

For the cross-project same-candidate artifact lane, every method must export
scores as:

```text
source_event_id,user_id,item_id,score
```

and final import/evaluation must use `main_import_same_candidate_baseline_scores.py`.
Do not use `test` split for hyperparameter selection. If reusing LLM2Rec
official results, reuse only scores/provenance/audit, not intermediate
checkpoint or embedding artifacts as long-term requirements.
