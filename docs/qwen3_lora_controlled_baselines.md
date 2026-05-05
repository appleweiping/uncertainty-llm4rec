# Qwen3-8B LoRA Controlled Baselines

This protocol defines the main fair-comparison lane for external LLM4Rec
projects. It is separate from official upstream reproduction.

## Principle

The main framework comparison should control the small LLM backbone and LoRA
budget. Otherwise, a result can reflect different base models, checkpoints,
tokenizers, or prompt vocabularies rather than the project framework.

Controlled main-table candidates must use:

- base model: `/home/ajifang/models/Qwen/Qwen3-8B`;
- tuning: LoRA or QLoRA with the shared config family below;
- dataset: canonical TRUCE processed splits;
- candidates: fixed TRUCE candidate sets with target inclusion unchanged;
- evaluator: TRUCE evaluator only;
- import: `candidate_scores.csv -> predictions.jsonl -> metrics.json`;
- split filter: `scripts/import_external_predictions.py --split test`.

Official upstream reproductions can still be useful, but they belong in a
separate reference/appendix table if they use T5, LLaMA, Vicuna, or a project
checkpoint not shared by the controlled comparison.

## First Two Controlled Baselines

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
```

Each controlled-baseline output directory contains:

- `train_sft.jsonl`
- `valid_sft.jsonl`
- `test_score_plan.jsonl`
- `controlled_baseline_manifest.json`
- `server_command_plan.md`

The server training runner must create:

- LoRA adapter checkpoint or checkpoint reference;
- `candidate_scores.csv`;
- stdout/stderr logs;
- environment and git info;
- imported TRUCE `predictions.jsonl`;
- TRUCE `metrics.json` and `metrics.csv`.

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

## Evidence Boundary

Zero-shot Qwen scoring, deterministic smoke scoring, and official-project
packets are not controlled LoRA results. A controlled baseline is complete only
after real LoRA training, candidate scoring, TRUCE import, and TRUCE evaluation.
