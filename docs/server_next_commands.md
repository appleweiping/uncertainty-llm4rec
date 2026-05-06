# Server Next Commands

This is the short handoff command sheet for continuing after the 2026-05-06
Main4 smoke completion.

## Pull Latest

```bash
cd ~/projects/TRUCE-Rec
git pull --ff-only
```

## Full Run Fast Controlled Baselines

Use the Qwen/torch/peft environment:

```bash
cd ~/projects/TRUCE-Rec
source ~/projects/TALLRec/.venv_tallrec/bin/activate

for NAME in \
  tallrec_qwen3_lora_amazon_beauty \
  dealrec_qwen3_lora_amazon_beauty \
  lc_rec_qwen3_lora_amazon_beauty
do
  echo "===== FULL $NAME ====="
  python scripts/run_qwen_lora_controlled_baseline.py \
    --manifest outputs/server_training/controlled_baselines/$NAME/controlled_baseline_manifest.json \
    --trust-remote-code
done
```

Do not full-run `openp5_style_qwen3_lora_amazon_beauty` yet. Its smoke passed,
but current scoring is too slow for full evaluation.

## Import And Evaluate

After a full run completes:

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

Only imported `predictions.jsonl` and TRUCE `metrics.json`/`metrics.csv` are
paper-eligible. Smoke metrics and raw project-side logs are not paper results.
