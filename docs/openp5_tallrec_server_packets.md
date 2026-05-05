# OpenP5 / TALLRec Server Packets

This document defines the next external-project baseline step. It contains no
experimental result claims.

For the broader external-project matrix covering BIGRec, DEALRec, LC-Rec,
LLaRA, CoLLM, LLM-ESR, and SLMRec, see
`docs/external_project_baseline_packets.md`.

## Source Repositories

- OpenP5 official repository: <https://github.com/agiresearch/OpenP5>
- TALLRec official repository: <https://github.com/SAI990323/TALLRec>

OpenP5 documents separate T5/LLaMA environments, a `generate_dataset.sh` data
generation step, training commands under `command/`, and evaluation commands
under `test_command/`. TALLRec documents `shell/instruct_7B.sh` for training and
`shell/evaluate.sh` for evaluation, with server-specific placeholders for
`base_model`, `train_data`, `val_data`, `test_data`, and output paths.

## Packet Generation

Prepare packets from canonical TRUCE processed data:

```powershell
py -3 scripts/prepare_project_baseline_packet.py --config configs/server/project_baselines/openp5_movielens_packet.yaml
py -3 scripts/prepare_project_baseline_packet.py --config configs/server/project_baselines/openp5_amazon_beauty_packet.yaml
py -3 scripts/prepare_project_baseline_packet.py --config configs/server/project_baselines/tallrec_movielens_packet.yaml
py -3 scripts/prepare_project_baseline_packet.py --config configs/server/project_baselines/tallrec_amazon_beauty_packet.yaml
```

Each packet writes:

- `project_baseline_manifest.json`
- `truce_examples.jsonl`
- `truce_candidate_sets.jsonl`
- `item_catalog.jsonl`
- project-facing train/valid/test files
- `candidate_scores_template.csv`

The external project must return `candidate_scores.csv` with columns:

```text
example_id,user_id,item_id,score
```

Then import and evaluate with TRUCE:

```powershell
py -3 scripts/import_external_predictions.py --scores <candidate_scores.csv> --examples <packet>/truce_examples.jsonl --output <run_dir>/predictions.jsonl --method openp5_official --source-project OpenP5 --model-name OpenP5 --seed 13 --split test
py -3 scripts/import_external_predictions.py --scores <candidate_scores.csv> --examples <packet>/truce_examples.jsonl --output <run_dir>/predictions.jsonl --method tallrec_official --source-project TALLRec --model-name TALLRec --seed 13 --split test
py -3 scripts/evaluate_predictions.py --predictions <run_dir>/predictions.jsonl --output-dir <run_dir>
```

Use `--split test` on `scripts/import_external_predictions.py` when importing
from a full `truce_examples.jsonl` packet. This preserves the final evaluation
split and avoids importing train/valid examples into diagnostic runs.

Do not use OpenP5/TALLRec evaluator metrics as paper metrics.

## Current Diagnostic Result

OpenP5 Amazon Beauty has one server-side adapter smoke row:

- method: `openp5_adapter_smoke_no_model`;
- run directory: `outputs/runs/openp5_adapter_smoke_amazon_beauty_seed13`;
- input scores:
  `~/projects/OpenP5/truce_outputs/openp5_beauty_adapter_smoke_candidate_scores.csv`;
- TRUCE evaluator count: 225 test examples;
- Recall@10 0.017778, NDCG@10 0.005872, MRR@10 0.002519.

This row verifies only the OpenP5 packet-to-TRUCE import/evaluation plumbing.
It uses deterministic no-model scores and is not an official OpenP5 result.
Do not include it in paper result tables.

Amazon Beauty has one server-side diagnostic row:

- method: TALLRec-style Qwen3 zero-shot scorer;
- run directory: `outputs/runs/tallrec_qwen_zeroshot_amazon_beauty_seed13`;
- Recall@10 0.031111, NDCG@10 0.011949, MRR@10 0.006321;
- final metrics computed by TRUCE evaluator on 225 test examples.

This row is an appendix/diagnostic result only. It is not an official trained
TALLRec baseline because TALLRec LoRA training was not run.

## OpenP5 Adapter Contract

Use the packet's `openp5/*_sequential_tasks.jsonl` as the bridge input to the
OpenP5 data/template layer. Preserve:

- canonical train/valid/test split;
- chronological history only;
- shared TRUCE candidates for scoring;
- target inclusion exactly as recorded in the packet manifest.

OpenP5 may generate item IDs/tokens or scores. The server adapter must ground
generated outputs back to canonical `item_id`. Invalid or ungrounded generations
must not be silently removed; they should receive low/zero score or be recorded
in metadata for TRUCE validity/hallucination auditing.

## TALLRec Adapter Contract

Use the packet's `tallrec/train.json`, `tallrec/valid.json`, and
`tallrec/test.json` as pairwise Yes/No data. The model input contains only
history and one candidate item. The `output` field is a label for training or
offline scoring and must never be included in the model prompt.

TALLRec's official evaluation script computes AUC and writes an aggregate JSON.
For TRUCE, add a server-side dump of the per-row Yes probability/logit and join
it with `tallrec/test_row_map.jsonl` to produce `candidate_scores.csv`.

## Fairness Rules

- Do not train on test rows.
- Do not change candidate sets.
- Do not tune Amazon Video Games.
- Do not copy official evaluator metrics into paper tables.
- Record upstream commit, environment, GPU, checkpoint, command, and logs in
  `project_baseline_manifest.json`.

## Blocker Policy

If OpenP5 or TALLRec cannot run on the server, report the concrete blocker:
missing checkpoint, CUDA/driver issue, package conflict, data conversion failure,
out-of-memory, or output-grounding mismatch. Do not substitute a mock result.
