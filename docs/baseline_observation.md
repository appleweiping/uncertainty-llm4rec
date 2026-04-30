# Baseline Observation Interface

Baseline observation is the first reviewer-proofing layer after API and mock
observation scaffolds. It keeps the same title-level file contract:

```text
observation inputs
  -> baseline raw/parsed predictions
  -> generated title grounding
  -> grounded predictions
  -> metrics/report/manifest
```

This is not heavy recommender training and not a paper result. It exists so the
pipeline can compare DeepSeek/API observations with deterministic non-LLM
baselines under the same grounding and confidence-analysis schema.

## Implemented Baselines

- `popularity`: recommends the most popular catalog title not already in the
  user's history. The confidence proxy is normalized log popularity.
- `cooccurrence`: builds item-to-next-item counts from `split=train`
  observation examples only, aggregates counts from the user's history with a
  recency weight, and falls back to popularity when no train co-occurrence is
  available.
- `ranking_jsonl`: converts externally produced ranked item IDs into the same
  title-level observation schema. This is the adapter contract for later
  SASRec, BERT4Rec, GRU4Rec, LightGCN, P5/TIGER/BIGRec-like, or other
  non-API baselines.

All implemented baseline paths emit or select an item title, not only a
ranked-list metric. The generated title is grounded back to the catalog before
correctness, GroundHit, ECE, Brier, head/mid/tail summaries, and risk counts
are computed.

## Command

Run on any existing observation input JSONL:

```powershell
python scripts/run_baseline_observation.py --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline popularity --max-examples 30
python scripts/run_baseline_observation.py --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline cooccurrence --max-examples 30
python scripts/run_baseline_observation.py --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline ranking_jsonl --ranking-jsonl outputs/baseline_rankings/sasrec/sample_5k/test_rankings.jsonl --max-examples 30 --strict-ranking
```

Outputs are ignored under:

```text
outputs/observations/baselines/<dataset>/<processed_suffix>/<input_stem>_<baseline>/
```

Files:

- `raw_responses.jsonl`
- `parsed_predictions.jsonl`
- `grounded_predictions.jsonl`
- `metrics.json`
- `report.md`
- `manifest.json`

## Guardrails

- No API is called.
- No API key is read.
- No model is trained.
- Co-occurrence counts are built from train-split observation examples only.
- `ranking_jsonl` consumes predictions produced elsewhere; this command only
  adapts ranked item IDs into grounded title predictions.
- `ranking_jsonl --strict-ranking` fails when an input has no usable ranking
  row. Without strict mode, missing or unusable ranking rows fall back to the
  popularity baseline and mark `baseline_fallback_reason`.
- Confidence is a simple proxy, not calibrated model confidence.
- Outputs are sanity/baseline artifacts until a full protocol run is explicitly
  executed and documented.

## Ranking JSONL Contract

The ranking adapter requires one JSON object per observation `input_id`. It
does not read the target item, and it filters already-seen history items before
selecting the top catalog item.

Supported shapes:

```json
{"input_id": "dataset:run:example:hash", "ranked_item_ids": ["item1", "item2"], "scores": [3.2, 1.1]}
{"input_id": "dataset:run:example:hash", "ranked_items": [{"item_id": "item1", "score": 3.2}]}
{"input_id": "dataset:run:example:hash", "recommendations": [{"item_id": "item1", "score": 3.2}]}
```

The adapter writes `baseline_selected_rank`, `baseline_candidate_count`,
`baseline_score`, `baseline_score_source`, and `baseline_source_record_id` into
parsed and grounded outputs. Confidence is a rank/score proxy used only to keep
the common analysis schema executable; it must not be described as calibrated
model confidence unless a later calibration stage validates it.

## Artifact Validation Gate

Before an externally trained ranker artifact is adapted through
`ranking_jsonl`, validate the source run manifest and then validate the local
ranking JSONL against the observation input slice:

```powershell
python scripts/validate_baseline_run_manifest.py --manifest-json runs/baselines/sasrec/amazon_reviews_2023_beauty/full/run_manifest.json --strict
python scripts/validate_baseline_artifact.py --ranking-jsonl outputs/baseline_rankings/sasrec/sample_5k/test_rankings.jsonl --input-jsonl outputs/observation_inputs/amazon_reviews_2023_beauty/sample_5k/test_forced_json.jsonl --baseline-family sasrec --model-family SASRec --run-label sasrec_sample_5k_test --dataset amazon_reviews_2023_beauty --processed-suffix sample_5k --split test --trained-splits train --strict
```

The source run manifest uses `baseline_ranking_run_manifest_v1`; a template is
available at `configs/server/baseline_ranking_run_manifest.example.json`. The
validator writes an ignored manifest under
`outputs/baseline_run_manifest_validation/...` and checks required provenance,
train/evaluation split separation, command/git/seed metadata, declared input
and ranking artifact paths, and the flags
`grounding_required_before_correctness=true` and
`uses_heldout_targets_for_training=false`.

The artifact validator writes an ignored manifest under
`outputs/baseline_artifact_validation/...` by default. It checks input coverage,
duplicate `input_id` values, candidate-list schema, score length/type, catalog
item-id coverage, already-seen history items, split/dataset declarations, and
basic provenance such as upstream config/source manifests when supplied. It
does not execute the ranker, call APIs, train a model, download data, or turn
the ranking file into a paper result.

Use `--allow-missing-inputs` only for an explicitly documented diagnostic. Use
`--fail-on-extra-rankings` when the ranking artifact should exactly match the
selected input slice. Extra ranking rows are otherwise treated as a warning so
a full ranking artifact can still be validated on a smaller observation slice.

## Next Extensions

Later baseline phases should plug trained ranking or generative baseline
artifacts into this adapter or an equivalent schema-compatible writer. Each
must write the same grounded prediction schema before entering analysis.
