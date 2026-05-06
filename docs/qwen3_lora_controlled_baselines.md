# Qwen3-8B LoRA Controlled Baselines

This protocol defines the main fair-comparison lane for external LLM4Rec
projects. It is separate from official upstream reproduction.

## Principle

The main framework comparison should control the small LLM backbone and LoRA
budget. Otherwise, a result can reflect different base models, checkpoints,
tokenizers, or prompt vocabularies rather than the project framework.

This suite is specifically for comparing TRUCE/CU-GR against reference-paper
LLM4Rec project families recommended for the paper baseline set. The goal is
not merely to reproduce their original checkpoints; it is to test whether our
framework remains stronger when each project is adapted to the same Qwen3-8B
LoRA backbone and the same TRUCE data/evaluator contract.

Controlled main-table candidates must use:

- base model: `/home/ajifang/models/Qwen/Qwen3-8B`;
- tuning: LoRA or QLoRA with the shared config family below;
- dataset: canonical TRUCE processed splits;
- candidates: fixed TRUCE candidate sets with target inclusion unchanged;
- evaluator: TRUCE evaluator only;
- import: `candidate_scores.csv -> predictions.jsonl -> metrics.json`;
- split filter: `scripts/import_external_predictions.py --split test`.

In addition to ranking metrics, each controlled baseline should be fed into the
observation analysis layer when possible. We need to check whether the
observation phenomena motivating TRUCE/CU-GR also appear in TALLRec/OpenP5/
DEALRec/LC-Rec style baselines, instead of only showing the phenomenon on a
weak base model.

Official upstream reproductions can still be useful, but they belong in a
separate reference/appendix table if they use T5, LLaMA, Vicuna, or a project
checkpoint not shared by the controlled comparison.

## Main4 Controlled Baselines

The first main-table suite should use four baselines:

| Baseline | Comparison role | Status |
| --- | --- | --- |
| TALLRec-Qwen3-LoRA | instruction tuning for recommendation | configured |
| OpenP5-style-Qwen3-LoRA | P5-style generative/sequential recommendation | configured |
| DEALRec-Qwen3-LoRA | data-efficient LLM4Rec | configured |
| LC-Rec-Qwen3-LoRA | LLM plus collaborative signal | configured |

These four cover the core external-framework families without spreading the
first experiment phase too thin. CoLLM/LLaRA should follow as additional
collaborative-signal baselines. LLM-ESR/SLMRec should follow as long-tail or
sequential-specialist robustness baselines.

The current Amazon Beauty processed set is useful for pipeline and early
comparison, but it may still be too small for final top-conference claims. When
the larger same-server dataset from the parallel data project is ready, rerun
the same controlled suite without changing the evaluator or candidate protocol.

### TALLRec-Qwen3-LoRA

- Config:
  `configs/server/controlled_baselines/tallrec_qwen3_lora_amazon_beauty.yaml`
- Packet:
  `outputs/server_packets/tallrec_amazon_beauty`
- Training input:
  pairwise Yes/No instruction rows from the TALLRec packet.
- Scoring:
  compute Yes likelihood/logit for every test candidate, then write
  `candidate_scores.csv`.
- Paper label after completion:
  `TALLRec-Qwen3-LoRA (controlled)`.

### OpenP5-Style-Qwen3-LoRA

- Config:
  `configs/server/controlled_baselines/openp5_style_qwen3_lora_amazon_beauty.yaml`
- Packet:
  `outputs/server_packets/openp5_amazon_beauty`
- Training input:
  OpenP5-style sequential prompts using mapped item tokens.
- Scoring:
  compute causal-LM likelihood for each candidate item token, then write
  `candidate_scores.csv`.
- Paper label after completion:
  `OpenP5-style-Qwen3-LoRA (controlled)`.

This is not an official OpenP5 T5/P5 result. It is the controlled-backbone
adaptation needed for a fair framework comparison.

### DEALRec-Qwen3-LoRA

- Config:
  `configs/server/controlled_baselines/dealrec_qwen3_lora_amazon_beauty.yaml`
- Packet:
  `outputs/server_packets/dealrec_amazon_beauty`
- Training input:
  pairwise data-efficient project-style prompts from the generic packet, with
  the target plus fixed sampled negatives.
- Scoring:
  compute `Yes.` likelihood for every TRUCE candidate prompt.
- Paper label after completion:
  `DEALRec-Qwen3-LoRA (controlled)`.

### LC-Rec-Qwen3-LoRA

- Config:
  `configs/server/controlled_baselines/lc_rec_qwen3_lora_amazon_beauty.yaml`
- Packet:
  `outputs/server_packets/lc_rec_amazon_beauty`
- Training input:
  pairwise collaborative-signal prompts using user history and candidate item
  text, with the target plus fixed sampled negatives.
- Scoring:
  compute `Yes.` likelihood for every TRUCE candidate prompt.
- Paper label after completion:
  `LC-Rec-Qwen3-LoRA (controlled)`.

## Prepare Server Inputs

On the server, after pulling the latest repository:

```bash
cd ~/projects/TRUCE-Rec
source .venv_truce/bin/activate

python scripts/prepare_project_baseline_packet.py \
  --config configs/server/project_baselines/tallrec_amazon_beauty_packet.yaml
python scripts/prepare_project_baseline_packet.py \
  --config configs/server/project_baselines/openp5_amazon_beauty_packet.yaml

python scripts/prepare_qwen_lora_controlled_baseline.py \
  --config configs/server/controlled_baselines/tallrec_qwen3_lora_amazon_beauty.yaml
python scripts/prepare_qwen_lora_controlled_baseline.py \
  --config configs/server/controlled_baselines/openp5_style_qwen3_lora_amazon_beauty.yaml
python scripts/prepare_qwen_lora_controlled_baseline.py \
  --config configs/server/controlled_baselines/dealrec_qwen3_lora_amazon_beauty.yaml
python scripts/prepare_qwen_lora_controlled_baseline.py \
  --config configs/server/controlled_baselines/lc_rec_qwen3_lora_amazon_beauty.yaml
```

Or prepare the full first suite:

```bash
python scripts/prepare_controlled_baseline_suite.py
```

Each controlled-baseline output directory contains:

- `train_sft.jsonl`
- `valid_sft.jsonl`
- `test_score_plan.jsonl`
- `controlled_baseline_manifest.json`
- `server_command_plan.md`

`prepare_controlled_baseline_suite.py` also writes a smoke run queue:

```text
outputs/server_training/controlled_baselines/qwen3_lora_main4_amazon_beauty/server_run_queue.sh
```

Run that queue first to validate the four training/scoring paths with tiny
limits, then remove `--max-*` flags for the full runs.

The server training runner must create:

- LoRA adapter checkpoint or checkpoint reference;
- `candidate_scores.csv`;
- stdout/stderr logs;
- environment and git info;
- imported TRUCE `predictions.jsonl`;
- TRUCE `metrics.json` and `metrics.csv`.

For a paper-eligible controlled baseline, also preserve enough raw artifacts to
run observation diagnostics:

- candidate score distributions;
- selected top candidates and target positions;
- prompt/source template metadata;
- latency/runtime summary;
- enough per-example metadata to slice long-tail, validity, and observation
  phenomena by bucket.

## Shared LoRA Budget

Initial budget:

- epochs: 1;
- batch size: 1;
- gradient accumulation: 16;
- learning rate: 2e-4;
- warmup ratio: 0.03;
- LoRA rank: 16;
- LoRA alpha: 32;
- dropout: 0.05;
- target modules:
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`;
- dtype: bf16;
- gradient checkpointing: true.

If memory fails, change the budget consistently across controlled baselines and
record the reason in each manifest. Do not silently lower one baseline only.

## Current Execution Status

As of 2026-05-06, all four Main4 controlled baselines have completed server
smoke runs with `--max-train-examples 128`, `--max-steps 5`, and
`--max-score-rows 2`.

- TALLRec-Qwen3-LoRA: smoke completed.
- OpenP5-style-Qwen3-LoRA: smoke completed but scoring is currently too slow
  for full run without optimization.
- DEALRec-Qwen3-LoRA: smoke completed after changing generic prompts to
  pairwise `Yes.` likelihood.
- LC-Rec-Qwen3-LoRA: smoke completed after changing generic prompts to pairwise
  `Yes.` likelihood.

Next full-run order:

1. TALLRec-Qwen3-LoRA.
2. DEALRec-Qwen3-LoRA.
3. LC-Rec-Qwen3-LoRA.
4. OpenP5-style-Qwen3-LoRA only after scoring optimization.

## Evidence Boundary

Zero-shot Qwen scoring, deterministic smoke scoring, and official-project
packets are not controlled LoRA results. A controlled baseline is complete only
after real LoRA training, candidate scoring, TRUCE import, and TRUCE evaluation.
